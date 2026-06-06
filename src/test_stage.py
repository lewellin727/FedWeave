"""Federated multi-silo test stage (single dataset) — deployment scenario.

Pipeline (test mode runs on a single (dataset, type) pair):
  offline:   ensure every test passage has a doc-LoRA on disk (silo-local training).
  online:    pool test passages, split into `num_silos` silos; each silo runs
             ColBERT MaxSim local retrieval; server runs coverage-greedy global
             selection; CAA forward aggregates the selected per-doc LoRAs to
             generate the answer.
  aggregate: merge worker shards into final result.json + eval.json.

Stages:
  --stage all        run offline then online (then aggregate iff num_workers == 1)
  --stage offline    only ensure doc-LoRAs (parallel-friendly)
  --stage online     only run federated inference (assumes LoRAs exist)
  --stage aggregate  only merge shards (single process; auto-globs all shards)
"""
import os
import glob
import json

import torch

from src.utils import get_model, evaluate, shard_list, shard_indices
from src.inference import caa_inference
from src.lora import train_lora
from src.caa import CAAModule, mount_caa, unmount_caa
from src.r_matrix import ColBERTEncoder
from src.dataset import load_aug_entries, iter_docs, entry_to_sample, make_doc_id
from src.silo import ColBertSilo, coverage_greedy_select


def test_stage(root_dir, dataset_name, dataset_type, model_name, augment_model, config, k=5, silo_k=None, gold_only=False,
               num_silos=6, worker_id=0, num_workers=1, stage='all', caa_eval_tag=None):
    """Federated test scenario on dataset/test/{dataset_name}/{dataset_type}/.

    Online stage loads the CAA module from config['caa']['train']['save_path']
    and uses CAA forward for LoRA aggregation (alpha read from config['caa']['alpha']).
    """
    assert stage in ('all', 'offline', 'online', 'aggregate'), f'test_stage: stage must be in {{all, offline, online, aggregate}}, got {stage}'
    save_dir = config['train']['save_dir']
    le = config['train']['lora_epoch']
    lora_save_dir = os.path.join(save_dir, f'{dataset_name}/{dataset_type}/{model_name}/test/le={le}/doc_lora')
    caa_subdir = caa_eval_tag if caa_eval_tag is not None else 'caa'
    eval_subdir = 'gold_only' if gold_only else f'silo_K={num_silos}_k={k}'
    eval_dir = os.path.join(save_dir, f'{dataset_name}/{dataset_type}/{model_name}/test/le={le}/{caa_subdir}/{eval_subdir}')

    if stage in ('all', 'offline'):
        _ensure_doc_loras(root_dir, dataset_name, dataset_type, model_name, augment_model, config, lora_save_dir, gold_only, worker_id, num_workers)

    if stage in ('all', 'online'):
        _federated_inference(root_dir, dataset_name, dataset_type, model_name, augment_model, config, lora_save_dir, eval_dir, k, silo_k, gold_only, num_silos, worker_id, num_workers)

    if stage == 'aggregate' or (stage == 'all' and num_workers == 1):
        _aggregate_shards(eval_dir)


# ------------------------------------------------------------------------------
# Stage A: ensure every test passage has a doc-LoRA on disk
# ------------------------------------------------------------------------------

def _ensure_doc_loras(root_dir, dataset_name, dataset_type, model_name, augment_model, config, lora_save_dir, gold_only, worker_id, num_workers):
    max_entries = config.get('data', {}).get('max_entries', None)
    entries = load_aug_entries(root_dir, dataset_name, dataset_type, max_entries, mode='test')
    if gold_only:
        pairs = [(pid, doc) for pid, doc in iter_docs(entries, augment_model) if doc.label == 'gold']
    else:
        pairs = list(iter_docs(entries, augment_model))

    # Shard the FULL (deterministic) pair list — NOT a runtime-snapshot of "missing".
    # The old code computed missing at each worker's startup; with N workers racing,
    # the snapshot differed across workers, so shards overlapped/skipped some pids.
    # Now: dedupe by pid + sort (stable across workers), shard, then per-item
    # existence check (skip if some other worker already wrote it).
    seen, unique = set(), []
    for pid, doc in pairs:
        if pid in seen: continue
        seen.add(pid); unique.append((pid, doc))
    pairs = sorted(unique, key=lambda p: p[0])
    my_assignment = shard_list(pairs, worker_id, num_workers)
    print(f'[w{worker_id}/{num_workers}] full pair list = {len(pairs)}; this worker handles {len(my_assignment)} (skip-if-exists).')
    if not my_assignment:
        return

    model, tokenizer, _ = get_model(model_name)
    trained = 0
    for local_idx, (pid, doc) in enumerate(my_assignment):
        doc_id = make_doc_id(dataset_name, dataset_type, pid)
        if os.path.exists(os.path.join(lora_save_dir, doc_id)):
            continue
        trained += 1
        print(f'\n[w{worker_id}/{num_workers}] [{local_idx + 1}/{len(my_assignment)}] training {doc_id}')
        train_lora(doc, doc_id, model, tokenizer, lora_save_dir, config)
    print(f'[w{worker_id}/{num_workers}] done: trained {trained} new LoRAs (skipped {len(my_assignment) - trained} already-existing)')


# ------------------------------------------------------------------------------
# Stage B: federated retrieval + LoRA merge + generate
# ------------------------------------------------------------------------------

def _federated_inference(root_dir, dataset_name, dataset_type, model_name, augment_model, config, lora_save_dir, eval_dir, k, silo_k, gold_only, num_silos, worker_id, num_workers):
    # Per-silo retrieval k. None or unset -> use select-k (current default; back-compat with main sweeps).
    if silo_k is None:
        silo_k = k
    print(f'[w{worker_id}/{num_workers}] per-silo retrieve k={silo_k}; server coverage-greedy select k={k}')
    max_entries = config.get('data', {}).get('max_entries', None)
    entries = load_aug_entries(root_dir, dataset_name, dataset_type, max_entries, mode='test')
    pid_to_doc = {pid: doc for pid, doc in iter_docs(entries, augment_model)}

    silo_path = os.path.join(root_dir, f'dataset/test/{dataset_name}/{dataset_type}/silo.json')
    assert os.path.isfile(silo_path), f'silo.json missing: {silo_path}. Run scripts/build_test_silo_splits.py first.'
    silo_json = json.load(open(silo_path, 'r'))
    assert len(silo_json) == num_silos, f'silo.json has {len(silo_json)} silos but num_silos={num_silos}'

    # Single ColBERT encoder shared by retrieval (ColBertSilo) and CAA (R-matrix).
    caa_cfg = config['caa']
    encoder = ColBERTEncoder(caa_cfg['colbert_path'], device='cuda:0', dtype=torch.float32)

    silos = []
    total_docs = total_skipped_no_aug = total_skipped_no_lora = 0
    silo_summary = []
    colbert_cache_dir = os.path.join(os.path.dirname(lora_save_dir), 'colbert_doc_emb')
    for sid in range(num_silos):
        kept_docs, kept_gids = [], []
        n_no_aug = n_no_lora = 0
        for pid in silo_json[str(sid)]:
            doc = pid_to_doc.get(pid)
            if doc is None or doc.augment is None or (isinstance(doc.augment, list) and len(doc.augment) == 0):
                n_no_aug += 1; continue
            gid = make_doc_id(dataset_name, dataset_type, pid)
            if not os.path.exists(os.path.join(lora_save_dir, gid)):
                n_no_lora += 1; continue
            kept_docs.append(doc); kept_gids.append(gid)
        total_docs += len(kept_docs); total_skipped_no_aug += n_no_aug; total_skipped_no_lora += n_no_lora
        silo_summary.append(f's{sid}={len(kept_docs)}(-{n_no_aug}aug,-{n_no_lora}lora)')
        if kept_docs:
            silos.append(ColBertSilo(sid, kept_docs, kept_gids, encoder, colbert_cache_dir, max_doc_len=caa_cfg.get('R_max_doc_len', 256)))
    print(f'[w{worker_id}/{num_workers}] silo.json loaded: {silo_path}')
    print(f'[w{worker_id}/{num_workers}] silos initialized: ' + ', '.join(silo_summary))
    print(f'[w{worker_id}/{num_workers}] total usable docs: {total_docs} (skipped {total_skipped_no_aug} no-aug, {total_skipped_no_lora} no-lora)')

    gid_to_silo = {gid: s.id for s in silos for gid in s.doc_global_ids}
    gid_to_passage = {gid: doc.passage for s in silos for gid, doc in zip(s.doc_global_ids, s.docs)}

    base_model, tokenizer, generation_config = get_model(model_name, device_map={'': 'cuda:0'})

    ckpt_path = caa_cfg['train']['save_path'].replace('.pt', '_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = caa_cfg['train']['save_path']
    assert os.path.exists(ckpt_path), f'CAA ckpt not found: {ckpt_path}. Train CAA first.'
    alpha = float(caa_cfg.get('alpha', 0.1))
    caa_module = CAAModule(hidden_size=base_model.config.hidden_size, intermediate_size=base_model.config.intermediate_size, L_caa=caa_cfg['L_caa'], d_a=caa_cfg['d_a'], alpha=alpha).to(base_model.device)
    state_dict = torch.load(ckpt_path, map_location=base_model.device, weights_only=True)
    missing, unexpected = caa_module.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f'[w{worker_id}/{num_workers}] WARNING: ckpt has {len(unexpected)} unexpected keys not in current model (e.g. {unexpected[0]}); ignored.')
    if missing:
        print(f'[w{worker_id}/{num_workers}] WARNING: model has {len(missing)} keys not in ckpt (e.g. {missing[0]}); using random init for those.')
    caa_module.eval()
    mount_caa(base_model, caa_module, caa_cfg['L_caa'])
    print(f'[w{worker_id}/{num_workers}] CAA ckpt loaded from {ckpt_path}; alpha={alpha}; mounted on {len(caa_cfg["L_caa"])} layers.')

    my_indices = shard_indices(len(entries), worker_id, num_workers)
    print(f'\n[w{worker_id}/{num_workers}] federated inference on {len(my_indices)}/{len(entries)} questions...')

    shard_records = []
    for local_i, idx in enumerate(my_indices):
        sample = entry_to_sample(entries[idx], augment_model, dataset_name, dataset_type)
        question = sample['question']
        print(f'\n[w{worker_id}] {local_i + 1}/{len(my_indices)} (idx={idx}, qid={sample["qid"]})')

        if gold_only:
            gold_idx = sample['gold_indices']
            if not gold_idx:
                shard_records.append({'idx': idx, 'qid': sample['qid'], 'pred': '', 'answer': sample['answer'], 'selected_doc_ids': [], 'silos': []})
                continue
            all_gold_gids = [sample['doc_ids'][i] for i in gold_idx]
            selected_global_ids = [gid for gid in all_gold_gids if os.path.exists(os.path.join(lora_save_dir, gid))]
            if not selected_global_ids:
                shard_records.append({'idx': idx, 'qid': sample['qid'], 'pred': '', 'answer': sample['answer'], 'selected_doc_ids': [], 'silos': []})
                continue
        else:
            q_emb = encoder.encode([question], max_length=caa_cfg.get('R_max_query_len', 32))[0]
            pooled = []
            for s in silos:
                pooled.extend(s.retrieve_with_coverage(q_emb, k=silo_k))
            print(f'  retrieved {len(pooled)} (silo, doc, coverage) candidates from {len(silos)} silos (silo_k={silo_k})')
            min_gain = config.get('retrieval', {}).get('coverage_min_gain', None)
            sel = coverage_greedy_select(pooled, k, min_gain=min_gain)
            selected_global_ids = [gid for _, gid in sel]
            if len(selected_global_ids) < k:
                print(f'  early-stop: kept {len(selected_global_ids)}/{k} (min_gain={min_gain})')

        silo_origins = [gid_to_silo.get(gid, '?') for gid in selected_global_ids]
        lora_paths = [os.path.join(lora_save_dir, gid) for gid in selected_global_ids]
        selected_passages = [gid_to_passage[gid] for gid in selected_global_ids]
        pred = caa_inference(question, base_model, caa_module, lora_paths, selected_passages, encoder, caa_cfg, tokenizer, generation_config)

        shard_records.append({'idx': idx, 'qid': sample['qid'], 'pred': pred, 'answer': sample['answer'], 'selected_doc_ids': selected_global_ids, 'silos': silo_origins})
        print(f'  pred:   {pred}')
        print(f'  truth:  {sample["answer"]}')
        print(f'  silos:  {silo_origins}')

    unmount_caa(base_model)

    shard_dir = os.path.join(eval_dir, 'shards')
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f'shard_w{worker_id}_of{num_workers}.json')
    json.dump(shard_records, open(shard_path, 'w'), indent=4)
    print(f'\n[w{worker_id}/{num_workers}] wrote {len(shard_records)} records to {shard_path}')


# ------------------------------------------------------------------------------
# Stage C: merge shards into result.json + eval.json
# ------------------------------------------------------------------------------

def _aggregate_shards(eval_dir):
    shard_dir = os.path.join(eval_dir, 'shards')
    shard_files = sorted(glob.glob(os.path.join(shard_dir, 'shard_w*_of*.json')))
    if not shard_files:
        print(f'[aggregate] no shards found under {shard_dir}; nothing to do.'); return

    all_records = []
    for f in shard_files:
        recs = json.load(open(f))
        print(f'  {os.path.basename(f)}: {len(recs)} records')
        all_records.extend(recs)
    all_records.sort(key=lambda r: r['idx'])

    idxs = [r['idx'] for r in all_records]
    assert len(set(idxs)) == len(idxs), f'duplicate idx across shards in {shard_dir}'
    assert idxs == sorted(idxs)
    print(f'  total: {len(all_records)} records (idx range {idxs[0]} .. {idxs[-1]})')

    preds = [r['pred'] for r in all_records]
    answers = [r['answer'] for r in all_records]
    per_q = [evaluate(p, a) for p, a in zip(preds, answers)]
    avg = {kk: sum(float(d[kk]) for d in per_q) / len(per_q) for kk in per_q[0]}

    json.dump(preds, open(os.path.join(eval_dir, 'result.json'), 'w'), indent=4)
    json.dump(avg, open(os.path.join(eval_dir, 'eval.json'), 'w'), indent=4)
    print(f'\n=== Final eval (N={len(preds)}) ===')
    for kk, v in avg.items():
        print(f'  {kk}: {v:.4f}')
    print(f'\nSaved to {eval_dir}')
