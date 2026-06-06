"""Silo abstraction for federated retrieval (ColBERT MaxSim + coverage-greedy).

Each silo holds a private subset of documents. At inference, silos export
(doc_id, coverage_vector) pairs; the server picks the global top-k by greedily
maximizing the submodular coverage utility U(S) = Σ_i max_d C_d[i] (FedWeave
paper Sec. III-B).
"""
import os
import json
import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.stats import dirichlet


def split_into_silos(passages, doc_global_ids, num_silos, alpha, seed, embedding_model_path):
    """Topic-cluster + Dirichlet non-IID split.

    Returns dict[silo_id] = list of indices into passages/doc_global_ids.
    """
    from sentence_transformers import SentenceTransformer
    np.random.seed(seed)

    model = SentenceTransformer(embedding_model_path)
    features = model.encode(passages, normalize_embeddings=True, show_progress_bar=False)

    kmeans = KMeans(n_clusters=num_silos, random_state=seed, n_init=10)
    topic_labels = kmeans.fit_predict(features)

    silo_data = {i: [] for i in range(num_silos)}
    for topic_id in range(num_silos):
        topic_indices = [i for i, label in enumerate(topic_labels) if label == topic_id]
        if not topic_indices:
            continue
        np.random.shuffle(topic_indices)
        proportions = dirichlet(alpha=np.repeat(alpha, num_silos)).rvs(1)[0]
        cut_points = (np.cumsum(proportions)[:-1] * len(topic_indices)).astype(int)
        splits = np.split(np.array(topic_indices), cut_points)
        for silo_id, idx_group in enumerate(splits):
            for idx in idx_group:
                silo_data[silo_id].append(int(idx))
    return silo_data


def load_or_compute_silo_split(passages, doc_global_ids, num_silos, alpha, seed, embedding_model_path, cache_path=None):
    """Compute silo split; if cache_path is given, reuse a previously written split."""
    if cache_path and os.path.isfile(cache_path):
        cached = json.load(open(cache_path, 'r'))
        if cached.get('num_silos') == num_silos and cached.get('seed') == seed and cached.get('alpha') == alpha and cached.get('n_docs') == len(passages):
            print(f'  silo split loaded from cache: {cache_path}')
            # Cached value is dict[str(silo_id) -> list of doc_global_ids]; convert back to indices.
            id_to_idx = {gid: i for i, gid in enumerate(doc_global_ids)}
            silo_data = {int(sid): [id_to_idx[gid] for gid in gids] for sid, gids in cached['assignment'].items()}
            return silo_data
    silo_data = split_into_silos(passages, doc_global_ids, num_silos, alpha, seed, embedding_model_path)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        snapshot = {'num_silos': num_silos, 'seed': seed, 'alpha': alpha, 'n_docs': len(passages),
                    'assignment': {str(sid): [doc_global_ids[i] for i in idxs] for sid, idxs in silo_data.items()}}
        json.dump(snapshot, open(cache_path, 'w'), indent=2)
        print(f'  silo split saved to {cache_path}')
    return silo_data


class ColBertSilo:
    """A silo storing ColBERT multi-vector embeddings per doc, serving coverage
    vector queries. Embeddings are cached on disk per (combo, doc).

    Communication model: the silo only exports (doc_id, coverage_vector) tuples
    upon retrieval. Raw passages never leave; matches FedRAG locality.

    Coverage vector C_d (Eq. 3): for each query token i, C_d[i] = max over doc
    tokens of <e_q^i, e_d^j>. Length = T_q (number of query tokens).
    """

    def __init__(self, silo_id, docs, doc_global_ids, encoder, cache_dir, max_doc_len=256):
        assert len(docs) == len(doc_global_ids)
        self.id = silo_id
        self.docs = docs
        self.doc_global_ids = doc_global_ids
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.max_doc_len = max_doc_len
        os.makedirs(cache_dir, exist_ok=True)
        embs = self._load_or_encode_embeddings()  # list of (T_d_i, 128) tensors
        # Vectorize: pad to (N, T_d_max, 128) so MaxSim retrieval becomes ONE batch
        # matmul instead of a Python for-loop over N docs (which dominated CPU before
        # — see issue diagnosed during 5-cell parallel sweep).
        N = len(embs)
        T_d_max = max((e.shape[0] for e in embs), default=1)
        self.doc_embs_padded = torch.zeros(N, T_d_max, 128, dtype=torch.float32)
        self.doc_mask = torch.zeros(N, T_d_max, dtype=torch.bool)
        for i, e in enumerate(embs):
            T = e.shape[0]
            self.doc_embs_padded[i, :T] = e
            self.doc_mask[i, :T] = True

    def _load_or_encode_embeddings(self):
        """Load each doc's (T_d, 128) tensor from cache; encode + cache misses in a batch."""
        cached, missing_idx, missing_passages = [None] * len(self.docs), [], []
        for i, gid in enumerate(self.doc_global_ids):
            p = os.path.join(self.cache_dir, f'{gid}.pt')
            if os.path.isfile(p):
                cached[i] = torch.load(p, map_location='cpu', weights_only=True)
            else:
                missing_idx.append(i)
                missing_passages.append(self.docs[i].passage)
        if missing_passages:
            print(f'  silo {self.id}: encoding {len(missing_passages)} doc embeddings (cache miss)')
            new_embs = self.encoder.encode(missing_passages, max_length=self.max_doc_len)
            for j, e in zip(missing_idx, new_embs):
                cached[j] = e
                torch.save(e, os.path.join(self.cache_dir, f'{self.doc_global_ids[j]}.pt'))
        return cached

    def retrieve_with_coverage(self, q_emb, k):
        """Return top-k local candidates ranked by MaxSim score together with
        their coverage vectors C_d (length T_q). Single batch matmul, no Python loop.

        Args:
            q_emb: (T_q, 128) cpu tensor, L2-normalized (from ColBERTEncoder.encode).
            k: how many local top candidates to return.
        Returns:
            List of (silo_id, doc_global_id, coverage_vector: (T_q,) float tensor).
        """
        # sims: (N, T_q, T_d_max) — broadcast q_emb across N docs
        sims = torch.einsum('qd,ntd->nqt', q_emb.float(), self.doc_embs_padded)
        # Mask padding tokens out of max
        sims = sims.masked_fill(~self.doc_mask.unsqueeze(1), float('-inf'))
        covs = sims.max(dim=-1).values   # (N, T_q) — coverage vectors
        scores = covs.sum(dim=-1)        # (N,)
        kk = min(k, scores.shape[0])
        top = torch.topk(scores, kk)
        return [(self.id, self.doc_global_ids[i.item()], covs[i.item()]) for i in top.indices]


MIN_KEEP_DOCS = 1  # server-side hard floor: never return fewer than this many docs (unless pool < MIN_KEEP_DOCS)


def coverage_greedy_select(candidates, k, min_gain=None):
    """Greedy selection on monotone submodular coverage utility U(S) = Σ_i max_d C_d[i].

    Theorem 1: greedy achieves (1 - 1/e) of optimum. Algorithm:
        S = ∅
        for k iterations:
            d* = argmax_{d ∉ S} (U(S ∪ {d}) - U(S))
            S = S ∪ {d*}

    Args:
        candidates: list of (silo_id, doc_id, coverage_vector) tuples (output of
                    ColBertSilo.retrieve_with_coverage pooled across silos).
        k: max number to select (hard cap).
        min_gain: optional float threshold on marginal coverage gain (sum over T_q
                  query tokens of ColBERT cosine, so units are "cosine-sum"). If the
                  next pick's gain is < min_gain, return early with fewer than k docs.
                  Floor is MIN_KEEP_DOCS — early-stop is only honored after that many
                  picks. So output size ∈ [min(MIN_KEEP_DOCS, len(candidates)), min(k, len(candidates))].
    Returns:
        List of (silo_id, doc_id) in order of selection.
    """
    if not candidates:
        return []
    T_q = candidates[0][2].shape[0]
    running = torch.zeros(T_q)  # U(∅) = 0; running[i] is max over chosen so far
    chosen, remaining = [], list(range(len(candidates)))
    for step in range(min(k, len(candidates))):
        best_gain, best_j = -float('inf'), -1
        for j in remaining:
            new_running = torch.maximum(running, candidates[j][2])
            gain = float((new_running - running).sum())
            if gain > best_gain:
                best_gain, best_j = gain, j
        if step >= MIN_KEEP_DOCS and min_gain is not None and best_gain < min_gain:
            break
        chosen.append(best_j)
        remaining.remove(best_j)
        running = torch.maximum(running, candidates[best_j][2])
    return [(candidates[j][0], candidates[j][1]) for j in chosen]
