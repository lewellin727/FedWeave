"""Data layer for FedHop: aug.json I/O, Document model, and sample builders.

This is the single canonical place for everything that knows the aug.json schema:
  - `Document`: per-passage object used by the LoRA trainer (passage + augments).
  - `make_doc_id`, `get_aug_path`, `load_aug_entries`: file paths and raw entries.
  - `iter_docs`, `entry_to_sample`: helpers for centralized/federated stages.
  - `iter_caa_samples` + resolvers: CAA-specific sample iterator (Phase 3).

aug.json schema (dataset/{mode}/{dataset}/{type}/aug.json):
    [
        {
            "qid": "...", "test_id": int,
            "question": str, "answer": str, "type": str,
            "passages": [{"id": int, "passage": str, "title": str,
                          "label": "gold"|"distractor", "supporting_sents": [...]}],
            "augments": [{"id": int,
                          "{augment_model}_rewrite": str,
                          "{augment_model}_qa": [...]}],
        },
        ...
    ]

Invariants:
- Within one (dataset, type) file: passage `id` is globally incrementing 0..N-1
- augments[i].id == passages[i].id (zip-friendly)
"""
import os
import json
import random

from datasets import Dataset as HFDataset

from src.utils import get_prompt


# ============================================================================
# Document: per-passage training object
# ============================================================================

class Document:
    def __init__(self, passage, aug_item, augment_model, title=None, label=None, supporting_sents=None):
        self.id = aug_item['id']
        self.passage = passage
        self.rewrite = aug_item[f'{augment_model}_rewrite']
        self.augment = aug_item[f'{augment_model}_qa']
        self.title = title
        self.label = label  # 'gold' | 'distractor' | None
        self.supporting_sents = supporting_sents

    def get_prompt_ids(self, tokenizer):
        prompt_ids = []
        if self.augment is None:
            return []
        qa_list_cnt = (len(self.augment) + 1) // 2
        for q_id, qa in enumerate(self.augment):
            if q_id < qa_list_cnt:
                for ppp in [self.passage, self.rewrite]:
                    prompt_ids.append(get_prompt(tokenizer, qa["question"], [ppp], qa["answer"]))
            else:
                prompt_ids.append(get_prompt(tokenizer, qa["question"], None, qa["answer"]))
        return prompt_ids

    def get_train_data(self, tokenizer, max_length=1024):
        prompt_ids = self.get_prompt_ids(tokenizer)
        pad_token_id = tokenizer.pad_token_id or 0
        eos_token_id = tokenizer.eos_token_id

        dataset = []
        for input_ids in prompt_ids:
            if eos_token_id is not None and input_ids[-1] != eos_token_id:
                input_ids = input_ids + [eos_token_id]
            labels = input_ids.copy()

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            padding_len = max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * padding_len
            input_ids += [pad_token_id] * padding_len
            labels += [-100] * padding_len

            dataset.append({"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})

        return HFDataset.from_list(dataset)


# ============================================================================
# Raw entry I/O
# ============================================================================

def make_doc_id(dataset, dataset_type, passage_id):
    """Globally unique, human-readable doc identifier."""
    return f'{dataset}_{dataset_type}_{passage_id}'


def get_aug_path(root_dir, dataset, dataset_type, mode='train'):
    """Return aug.json path under dataset/{mode}/{dataset}/{type}/."""
    return os.path.join(root_dir, 'dataset', mode, dataset, dataset_type, 'aug.json')


def load_aug_entries(root_dir, dataset, dataset_type, max_entries=None, mode='train'):
    """Load aug.json entries from dataset/{mode}/; optionally truncate."""
    path = get_aug_path(root_dir, dataset, dataset_type, mode)
    with open(path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    if max_entries is not None:
        entries = entries[:max_entries]
    return entries


# ============================================================================
# Stage-agnostic helpers (centralized train / federated test)
# ============================================================================

def iter_docs(entries, augment_model):
    """Yield (passage_id, Document) for every passage across all entries."""
    for entry in entries:
        for psg, aug in zip(entry['passages'], entry['augments']):
            assert psg['id'] == aug['id'], f"passage/augment id mismatch in qid={entry.get('qid')}: {psg['id']} vs {aug['id']}"
            doc = Document(passage=psg['passage'], aug_item=aug, augment_model=augment_model, title=psg.get('title'), label=psg.get('label'), supporting_sents=psg.get('supporting_sents'))
            yield psg['id'], doc


def entry_to_sample(entry, augment_model, dataset, dataset_type):
    """Convert one aug.json entry into the generic stage sample dict (used by test_stage)."""
    docs, doc_ids, gold_indices = [], [], []
    for i, (psg, aug) in enumerate(zip(entry['passages'], entry['augments'])):
        assert psg['id'] == aug['id']
        doc = Document(passage=psg['passage'], aug_item=aug, augment_model=augment_model, title=psg.get('title'), label=psg.get('label'), supporting_sents=psg.get('supporting_sents'))
        docs.append(doc)
        doc_ids.append(make_doc_id(dataset, dataset_type, psg['id']))
        if psg.get('label') == 'gold':
            gold_indices.append(i)
    return {'qid': entry.get('qid'), 'question': entry['question'], 'answer': entry['answer'], 'docs': docs, 'doc_ids': doc_ids, 'gold_indices': gold_indices}


# ============================================================================
# CAA training sample builder (Phase 3, C1 dynamic-N sampling)
# ============================================================================

def load_caa_training_entries(combos, root_dir='.', k_per_combo=50, mode='train'):
    """Load entries across multiple (dataset, type) combos, tagging each with origin."""
    out = []
    for ds, typ in combos:
        entries = load_aug_entries(root_dir=root_dir, dataset=ds, dataset_type=typ, max_entries=k_per_combo, mode=mode)
        for e in entries:
            out.append({'dataset': ds, 'dataset_type': typ, 'entry': e})
    return out


def gold_indices(entry):
    """Return positions in entry['passages'] whose label == 'gold'."""
    return [i for i, p in enumerate(entry['passages']) if p.get('label') == 'gold']


def usable_passage_indices(entry, augment_model):
    """Positions in entry['passages'] that have non-empty augments (i.e., a doc-LoRA can be trained)."""
    out = []
    for i, aug in enumerate(entry['augments']):
        qa = aug.get(f'{augment_model}_qa')
        if qa is None or (isinstance(qa, list) and len(qa) == 0):
            continue
        out.append(i)
    return out


def sample_n_docs(entry, augment_model, n_min=2, rng=None):
    """C1 sampling restricted to usable (non-empty-augment) passages.

    N uniform in [max(n_min, num_gold_usable), num_usable]. Returns None when
    the entry can't satisfy n_min (e.g. a gold doc had empty augments and
    we'd fall below the floor).
    """
    rng = rng or random
    usable = usable_passage_indices(entry, augment_model)
    if len(usable) < n_min:
        return None
    gold = [i for i in usable if entry['passages'][i].get('label') == 'gold']
    distractors = [i for i in usable if entry['passages'][i].get('label') != 'gold']
    num_gold = len(gold)
    lo = max(n_min, num_gold)
    if lo > len(usable):
        return None
    n = rng.randint(lo, len(usable))
    extra = rng.sample(distractors, n - num_gold)
    sampled = gold + extra
    rng.shuffle(sampled)
    return sampled


def build_caa_sample(entry, sampled_indices, augment_model, dataset, dataset_type):
    """Pack a sampled entry into a CAA training sample.

    Returns:
        {
          'dataset', 'dataset_type', 'qid', 'question', 'answer',
          'docs':            List[Document],          # len == N, ordered by sampled_indices
          'doc_ids':         List[str],               # global ids matching docs
          'doc_passage_ids': List[int],               # local passage 'id'
          'gold_positions':  List[int],               # positions in 'docs' that are gold
          'sampled_indices': List[int],               # original positions chosen (for R slicing)
          'all_passages':    List[str],               # full passage list (for caching full-R)
        }
    """
    docs, doc_ids, doc_pids, gold_positions = [], [], [], []
    for new_pos, orig_idx in enumerate(sampled_indices):
        psg = entry['passages'][orig_idx]
        aug = entry['augments'][orig_idx]
        assert psg['id'] == aug['id'], f"id mismatch at idx {orig_idx} in qid={entry.get('qid')}"
        doc = Document(passage=psg['passage'], aug_item=aug, augment_model=augment_model, title=psg.get('title'), label=psg.get('label'), supporting_sents=psg.get('supporting_sents'))
        docs.append(doc)
        doc_ids.append(make_doc_id(dataset, dataset_type, psg['id']))
        doc_pids.append(psg['id'])
        if psg.get('label') == 'gold':
            gold_positions.append(new_pos)
    all_passages = [p['passage'] for p in entry['passages']]
    return {'dataset': dataset, 'dataset_type': dataset_type, 'qid': entry.get('qid'), 'question': entry['question'], 'answer': entry['answer'], 'docs': docs, 'doc_ids': doc_ids, 'doc_passage_ids': doc_pids, 'gold_positions': gold_positions, 'sampled_indices': sampled_indices, 'all_passages': all_passages}


def iter_caa_samples(combos, augment_model, root_dir='.', k_per_combo=50, mode='train', n_min=2, seed=42):
    """Generator over CAA training samples across all combos with deterministic per-entry sampling.

    Entries whose usable (non-empty-augment) passages can't satisfy n_min are silently skipped.
    """
    tagged = load_caa_training_entries(combos, root_dir=root_dir, k_per_combo=k_per_combo, mode=mode)
    skipped = 0
    for item in tagged:
        ds, typ, entry = item['dataset'], item['dataset_type'], item['entry']
        key = f'{ds}|{typ}|{entry.get("qid")}|{seed}'
        rng = random.Random(hash(key) & 0xFFFFFFFF)
        sampled = sample_n_docs(entry, augment_model, n_min=n_min, rng=rng)
        if sampled is None:
            skipped += 1
            continue
        yield build_caa_sample(entry, sampled, augment_model, ds, typ)
    if skipped:
        print(f'[iter_caa_samples] skipped {skipped} entries with insufficient usable passages')


def get_doc_lora_paths(sample, save_dir, model_name, lora_epoch, mode='train'):
    """Resolve on-disk doc-LoRA paths for a CAA sample.

    Mirrors train_stage / test_stage's path scheme:
      {save_dir}/{ds}/{type}/{model}/{mode}/le={le}/doc_lora/{doc_id}
    """
    lora_root = os.path.join(save_dir, sample['dataset'], sample['dataset_type'], model_name, mode, f'le={lora_epoch}', 'doc_lora')
    return [os.path.join(lora_root, did) for did in sample['doc_ids']]


def get_R_for_sample(sample, encoder, caa_config, mode='train'):
    """Compute (or load cached) R and per-doc ColBERT scores, sliced to the sampled subset.

    Returns (R_sliced: float32 (N, N), scores_sliced: float32 (N,)). Cache key is
    full-doc-set per qid; slicing happens at runtime so dynamic N selections never
    invalidate the cache. scores feed the CAA router prior (alpha_logits += log_softmax(scores)).
    """
    from src.r_matrix import get_or_compute_R_and_scores
    R_full, scores_full = get_or_compute_R_and_scores(query=sample['question'], docs=sample['all_passages'], qid=sample['qid'], encoder=encoder, cache_dir=caa_config['R_cache_dir'], dataset=sample['dataset'], dataset_type=sample['dataset_type'], mode=mode, max_doc_len=caa_config['R_max_doc_len'], max_query_len=caa_config['R_max_query_len'])
    idx = sample['sampled_indices']
    return R_full[idx][:, idx].contiguous(), scores_full[idx].contiguous()


def assert_lora_paths_exist(sample, lora_paths):
    """Fail fast if any doc-LoRA dir is missing — points to incomplete offline run."""
    missing = [p for p in lora_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'qid={sample["qid"]} ({sample["dataset"]}/{sample["dataset_type"]}): {len(missing)}/{len(lora_paths)} doc-LoRAs missing — first: {missing[0]}')
