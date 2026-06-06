"""R-matrix and per-doc score computation via ColBERTv2.

Both signals come from one ColBERT pass over (query, docs):

  R[i, j] in [0, 1]  : query-aware doc-doc complementarity (1 - cosine on per-
                       query-token MaxSim vectors, with min-max normalization).
                       High R = docs cover the query differently (good for
                       multi-hop). Diagonal forced to 1.0. Used as `+ log(R)`
                       bias on the cross-LoRA attention.
  scores[i]          : per-doc ColBERT MaxSim score, summed over query tokens
                       (standard ColBERT scoring). Used as `+ log_softmax(scores)`
                       bias on the per-LoRA router.

Formula for R:
  1. ColBERT encode q, each d_i -> L2-normalized 128-dim token embeddings.
  2. Per doc_i: w_i[t] = max_s <q_t, d_i_s>   (MaxSim per query token, length T_q).
  3. L2-normalize each w_i. C[i,j] = w_i_norm · w_j_norm. R_raw = 1 - C.
  4. Min-max normalize off-diagonal to [R_FLOOR, 1.0] per question.
  5. R[i,i] := 1.0. R := clamp(R, 0, 1).
  scores[i] = w_i.sum() (computed from the same w_i in step 2, before normalization).

Disk cache (lazy, atomic writes via mkstemp + os.replace):
  <R_cache_dir>/<ds>/<type>/<mode>/<qid>.npy           # R         (N, N) float32
  <R_cache_dir>/<ds>/<type>/<mode>/<qid>.scores.npy    # scores    (N,)   float32
"""
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, BertModel


COLBERT_PROJ_DIM = 128
COLBERT_HIDDEN = 768

R_FLOOR = 0.05   # floor for off-diagonal R after min-max normalization;
                 # log(0.05) ~= -3.0 bounds the worst-case cross-attn penalty


# ============================================================================
# ColBERT encoder (HF colbertv2.0 = BERT-base + 768->128 linear projection)
# ============================================================================

class ColBERTEncoder(nn.Module):
    """Loads colbertv2.0 (HF format) and returns L2-normalized token embeddings."""

    def __init__(self, model_path, device='cuda', dtype=torch.float32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True, dtype=dtype, add_pooling_layer=False)
        self.linear = nn.Linear(COLBERT_HIDDEN, COLBERT_PROJ_DIM, bias=False)
        weights = load_file(os.path.join(model_path, 'model.safetensors'))
        assert 'linear.weight' in weights, f'colbert projection layer missing in {model_path}'
        self.linear.weight.data = weights['linear.weight'].to(dtype)
        self.eval().to(device)
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, texts, max_length=256):
        """Encode a list of strings; return list of (T_i, 128) cpu tensors (padding stripped)."""
        if not texts:
            return []
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(self.device)
        hidden = self.bert(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
        emb = self.linear(hidden)
        emb = F.normalize(emb, p=2, dim=-1)
        result = []
        for i in range(len(texts)):
            T = int(enc.attention_mask[i].sum().item())
            result.append(emb[i, :T].cpu())
        return result


# ============================================================================
# R-matrix computation
# ============================================================================

def compute_R_via_colbert(query, docs, encoder, max_doc_len=256, max_query_len=32):
    """Compute the (N, N) R matrix and the per-doc query-relevance scores for one query.

    Returns:
        R:      torch.float32 cpu tensor (N, N) in [0, 1]. Off-diagonal entries
                encode pairwise complementarity (1 - cosine sim on per-query-token
                MaxSim vectors). Diagonal forced to 1.0.
        scores: torch.float32 cpu tensor (N,). Per-doc ColBERT score (sum over
                query tokens of MaxSim with the doc). Used as a soft prior on
                the per-LoRA router (alpha_logits += log_softmax(scores)).
    """
    N = len(docs)
    if N == 0:
        return torch.zeros(0, 0), torch.zeros(0)
    q_emb = encoder.encode([query], max_length=max_query_len)[0]
    doc_embs = encoder.encode(docs, max_length=max_doc_len)

    w_list = [(q_emb @ d_emb.T).max(dim=-1).values for d_emb in doc_embs]
    W = torch.stack(w_list, dim=0)                  # (N, T_q)
    scores = W.sum(dim=-1).float().contiguous()     # (N,) ColBERT MaxSim score per doc

    if N == 1:
        return torch.ones(1, 1), scores

    W_norm = F.normalize(W, p=2, dim=-1)
    C = W_norm @ W_norm.T
    R = 1.0 - C

    eye = torch.eye(N, dtype=torch.bool)
    off = R[~eye]
    if off.numel() >= 1:
        lo, hi = off.min(), off.max()
        if (hi - lo) > 1e-6:
            R_scaled = R_FLOOR + (R - lo) / (hi - lo) * (1.0 - R_FLOOR)
            R = torch.where(eye, R, R_scaled)

    R.fill_diagonal_(1.0)
    R = R.clamp(0.0, 1.0)
    return R, scores


# ============================================================================
# Disk cache
# ============================================================================

def get_r_cache_path(cache_dir, dataset, dataset_type, mode, qid):
    """Cache path for one R matrix. qid is sanitized for filesystem safety."""
    safe_qid = str(qid).replace('/', '_')
    return os.path.join(cache_dir, dataset, dataset_type, mode, f'{safe_qid}.npy')


def get_scores_cache_path(cache_dir, dataset, dataset_type, mode, qid):
    """Sibling cache for the per-doc ColBERT scores used by the CAA router prior."""
    safe_qid = str(qid).replace('/', '_')
    return os.path.join(cache_dir, dataset, dataset_type, mode, f'{safe_qid}.scores.npy')


def load_R(cache_dir, dataset, dataset_type, mode, qid):
    """Return R as a float32 tensor, or None if not cached."""
    r_path = get_r_cache_path(cache_dir, dataset, dataset_type, mode, qid)
    if not os.path.exists(r_path):
        return None
    return torch.from_numpy(np.load(r_path)).float()


def load_scores(cache_dir, dataset, dataset_type, mode, qid):
    """Return per-doc scores as a float32 tensor, or None if not cached."""
    s_path = get_scores_cache_path(cache_dir, dataset, dataset_type, mode, qid)
    if not os.path.exists(s_path):
        return None
    return torch.from_numpy(np.load(s_path)).float()


def _atomic_save_npy(final_path, arr):
    """Write arr to a tempfile in the same dir, then os.replace -> atomic on POSIX.

    Concurrent writers either succeed or no-op; readers never see a partial file.
    The previous non-atomic np.save let parallel DDP jobs catch each other
    mid-write, surfacing as `EOFError: No data left in file` on load_scores/load_R.
    """
    d = os.path.dirname(final_path)
    os.makedirs(d, exist_ok=True)
    # tempfile.mkstemp gives an exclusive fd in the SAME dir so os.replace is atomic
    # (cross-filesystem rename would fall back to copy+unlink, defeating the point).
    fd, tmp = tempfile.mkstemp(prefix='.tmp_', suffix='.npy', dir=d)
    os.close(fd)
    try:
        np.save(tmp, arr.astype(np.float32))
        # np.save adds .npy if not present — handle both
        if not os.path.exists(tmp) and os.path.exists(tmp + '.npy'):
            tmp = tmp + '.npy'
        os.replace(tmp, final_path)
    except Exception:
        for cand in (tmp, tmp + '.npy'):
            if os.path.exists(cand):
                try: os.unlink(cand)
                except OSError: pass
        raise


def save_R(cache_dir, dataset, dataset_type, mode, qid, R):
    """Save R as .npy atomically under the cache hierarchy."""
    r_path = get_r_cache_path(cache_dir, dataset, dataset_type, mode, qid)
    arr = R.cpu().numpy() if torch.is_tensor(R) else np.asarray(R)
    _atomic_save_npy(r_path, arr)


def save_scores(cache_dir, dataset, dataset_type, mode, qid, scores):
    """Save per-doc scores atomically as .scores.npy sibling of R."""
    s_path = get_scores_cache_path(cache_dir, dataset, dataset_type, mode, qid)
    arr = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    _atomic_save_npy(s_path, arr)


def get_or_compute_R_and_scores(query, docs, qid, encoder, cache_dir, dataset, dataset_type, mode, **kwargs):
    """Lazy fetch (R, scores): hit cache, else compute via ColBERT and save both.

    R and scores are cached as independent sibling .npy files so either can be
    back-filled without invalidating the other.
    """
    R = load_R(cache_dir, dataset, dataset_type, mode, qid)
    scores = load_scores(cache_dir, dataset, dataset_type, mode, qid)
    if R is not None and scores is not None:
        return R, scores
    R_new, scores_new = compute_R_via_colbert(query, docs, encoder, **kwargs)
    if R is None:
        save_R(cache_dir, dataset, dataset_type, mode, qid, R_new)
        R = R_new
    if scores is None:
        save_scores(cache_dir, dataset, dataset_type, mode, qid, scores_new)
        scores = scores_new
    return R, scores
