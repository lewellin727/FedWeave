"""Server-side CLA pre-training stage.

Orchestrates the full pre-training pipeline over the combined train pool:
  offline:  per-doc LoRAs across all (dataset, type) combos (parallel-friendly)
  cla:      train the CLA module on combined-pool samples (single-process)
  all:      offline -> cla

Outputs land under config['train']['save_dir']:
  doc-LoRAs at  {save_dir}/{ds}/{type}/{model}/train/le={le}/doc_lora/{doc_id}
  CLA ckpt at   config['cla']['train']['save_path']

Consumed downstream by test_stage at deployment time.
"""
import os

from src.utils import get_model, shard_list
from src.lora import train_lora
from src.dataset import load_aug_entries, iter_docs, make_doc_id


def train_stage(root_dir, datasets, model_name, augment_model, config, gold_only=False, worker_id=0, num_workers=1, stage='all', ranks_per_gpu=1):
    """Run server CLA pre-training over the combined train pool.

    Args:
        datasets:        list of (dataset_name, dataset_type) tuples to combine.
        gold_only:       if True, only train LoRAs for passages labeled 'gold' (debug).
        worker_id / num_workers: round-robin shard across parallel processes (offline only).
        stage:           'offline' | 'cla' | 'all'.
        ranks_per_gpu:   for CLA DDP; passed through to cla_train. Ignored in offline.
    """
    assert datasets, 'datasets must be non-empty'
    assert stage in ('all', 'offline', 'cla'), f'train_stage: stage must be in {{all, offline, cla}}, got {stage}'

    if stage in ('all', 'offline'):
        _train_doc_loras(root_dir, datasets, model_name, augment_model, config, gold_only, worker_id, num_workers)

    if stage in ('all', 'cla'):
        # Under DDP we let cla_train run on every rank; otherwise gate to worker 0 only.
        if 'LOCAL_RANK' not in os.environ and num_workers > 1 and worker_id != 0:
            print(f'[w{worker_id}/{num_workers}] CLA sub-stage is single-process; only w0 runs it. exiting.')
            return
        _train_cla_module(datasets, model_name, augment_model, config, ranks_per_gpu=ranks_per_gpu)


# ------------------------------------------------------------------------------
# Sub-stage A: per-doc LoRA training over the combined train pool
# ------------------------------------------------------------------------------

def _train_doc_loras(root_dir, datasets, model_name, augment_model, config, gold_only, worker_id, num_workers):
    """Train one LoRA per passage across all (dataset, type) combos in `datasets`.

    Idempotent: train_lora skips any doc_id whose adapter directory already exists,
    so re-running this stage (e.g. after expanding max_entries) only fills gaps.
    """
    max_entries = config.get('data', {}).get('max_entries', None)
    save_dir = config['train']['save_dir']
    lora_epoch = config['train']['lora_epoch']

    flat = []  # (ds_name, ds_type, pid, doc) flattened across all datasets
    print(f'[w{worker_id}/{num_workers}] offline: loading train aug entries from {len(datasets)} (dataset, type) pair(s) ...')
    for ds_name, ds_type in datasets:
        entries = load_aug_entries(root_dir, ds_name, ds_type, max_entries, mode='train')
        if gold_only:
            pairs = [(pid, doc) for pid, doc in iter_docs(entries, augment_model) if doc.label == 'gold']
        else:
            pairs = list(iter_docs(entries, augment_model))
        print(f'  {ds_name}/{ds_type}: {len(entries)} entries -> {len(pairs)} passages')
        for pid, doc in pairs:
            flat.append((ds_name, ds_type, pid, doc))

    my_assignment = shard_list(flat, worker_id, num_workers)
    print(f'[w{worker_id}/{num_workers}] offline: combined pool {len(flat)} doc-LoRAs total; this worker handles {len(my_assignment)}.')
    if not my_assignment:
        return

    model, tokenizer, _ = get_model(model_name)
    for local_idx, (ds_name, ds_type, pid, doc) in enumerate(my_assignment):
        doc_id = make_doc_id(ds_name, ds_type, pid)
        lora_save_dir = os.path.join(save_dir, f'{ds_name}/{ds_type}/{model_name}/train/le={lora_epoch}/doc_lora')
        print(f'\n[w{worker_id}/{num_workers}] [{local_idx + 1}/{len(my_assignment)}] doc_id={doc_id} label={doc.label}')
        train_lora(doc, doc_id, model, tokenizer, lora_save_dir, config)


# ------------------------------------------------------------------------------
# Sub-stage B: CLA module training (single-process)
# ------------------------------------------------------------------------------

def _train_cla_module(datasets, model_name, augment_model, config, ranks_per_gpu=1):
    """Train CLAModule on samples from `datasets` (assumes doc-LoRAs exist on disk).

    Reads CLA training hyperparams from config['cla']['train']. Passes ranks_per_gpu
    through so cla_train can bind each DDP rank to the correct GPU.
    """
    import torch
    from src.cla_train import cla_train
    tc = config['cla']['train']
    dtype_str = tc.get('dtype', 'bf16').lower()
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(dtype_str, torch.bfloat16)
    cla_train(datasets, model_name, augment_model, config, num_epochs=tc['num_epochs'], lr=float(tc['lr']),
                grad_accum_steps=tc['grad_accum_steps'], weight_decay=tc.get('weight_decay', 0.01), log_every=tc.get('log_every', 10),
                val_every=tc.get('val_every', 50), val_ratio=tc.get('val_ratio', 0.1), k_per_combo=tc.get('k_per_combo', 50),
                seed=tc.get('seed', 42), save_path=tc.get('save_path'), dtype=dtype, ranks_per_gpu=ranks_per_gpu)
