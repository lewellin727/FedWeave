"""FedWeave entry point.

Thin CLI dispatcher. Each --mode delegates the full pipeline to its stage file.

  --mode train  -> src/train_stage.py
                   Server-side CAA pre-training over the combined train pool.
                   Sub-stages: offline (doc-LoRAs, parallel) | caa (CAA module)
                   | all (offline -> caa).

  --mode test   -> src/test_stage.py
                   Federated multi-silo deployment on one (dataset, type) pair.
                   Sub-stages: offline (per-silo doc-LoRAs) | online (federated
                   retrieval + CAA-aggregated generation) | aggregate | all.

Dataset selection:
  --datasets X:Y[,X:Y...]    Comma-separated dataset:type pairs.
                             train: any number; test: exactly one.
"""
import os
import yaml
import argparse


def parse_datasets(s):
    pairs = []
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert ':' in chunk, f'malformed datasets entry: {chunk!r} (expected "dataset:type")'
        ds, t = chunk.split(':', 1)
        pairs.append((ds.strip(), t.strip()))
    assert pairs, 'must provide at least one dataset:type pair'
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=['train', 'test'])
    parser.add_argument("--stage", type=str, default='all', choices=['all', 'offline', 'online', 'aggregate', 'caa'],
                        help='Sub-stage selection. '
                             'Train mode: "offline" trains doc-LoRAs (parallel); "caa" trains CAA module '
                             '(single-process, requires LoRAs to exist); "all" runs offline -> caa. '
                             'Test mode: "all" runs offline -> online (and aggregate if num_workers=1); '
                             '"aggregate" only merges shards.')
    parser.add_argument("--datasets", type=str, required=True,
                        help='Comma-separated list of dataset:type pairs. '
                             'train: any number; test: exactly one.')
    parser.add_argument("--model_name", type=str, default='llama3.2-1b-instruct', choices=['llama3.2-1b-instruct', 'llama3-8b-instruct'])
    parser.add_argument("--augment_model", type=str, default="llama3.2-1b-instruct")
    parser.add_argument("--num_silos", type=int, default=6, help='test mode: number of silos')
    parser.add_argument("--k", type=int, default=5, help='test mode: server-side coverage-greedy select-k (final docs to LM). Per-silo retrieval k defaults to this; override with --silo_k.')
    parser.add_argument("--silo_k", type=int, default=None, help='test mode: per-silo retrieval top-k (pool size = num_silos × silo_k before greedy select). Defaults to --k if not given.')
    parser.add_argument("--gold_only", action='store_true',
                        help='offline: only train gold-labeled LoRAs. '
                             'online: skip retrieval/silos, use each question own gold docs directly.')
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_entries", type=int, default=None,
                        help='Override config.data.max_entries: cap entries per (dataset, type) to first N.')
    parser.add_argument("--caa_alpha", type=float, help='Override caa.alpha from config: scalar multiplier on the cross-attn residual.')
    parser.add_argument("--caa_k_per_combo", type=int, help='Override caa.train.k_per_combo from config (training only).')
    parser.add_argument("--caa_num_epochs", type=int, help='Override caa.train.num_epochs from config (training only).')
    parser.add_argument("--caa_save_path", type=str, help='Override caa.train.save_path from config (train: where to save; test: which ckpt to load).')
    parser.add_argument("--caa_eval_tag", type=str, default=None, help='Test mode: override the caa_subdir under test/le=.../ so each run writes to its own eval dir.')
    parser.add_argument("--coverage_min_gain", type=float, default=None, help='Test mode: early-stop threshold on marginal coverage gain (cosine-sum over T_q query tokens). Overrides retrieval.coverage_min_gain. Always keeps ≥1 doc.')
    parser.add_argument("--ranks_per_gpu", type=int, default=1,
                        help='CAA DDP: with torchrun --nproc_per_node=N and CUDA_VISIBLE_DEVICES set to M GPUs, ranks 0..N-1 map to visible GPU idx = LOCAL_RANK // ranks_per_gpu (so N = M * ranks_per_gpu).')
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config = yaml.load(open(os.path.join(curr_dir, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    if args.max_entries is not None:
        config.setdefault('data', {})['max_entries'] = args.max_entries
        print(f'[main] overriding data.max_entries -> {args.max_entries}')
    caa_cfg = config.setdefault('caa', {})
    if args.caa_alpha is not None:
        caa_cfg['alpha'] = args.caa_alpha
        print(f'[main] overriding caa.alpha -> {args.caa_alpha}')
    caa_tc = caa_cfg.setdefault('train', {})
    for arg_name, cfg_key in [('caa_k_per_combo', 'k_per_combo'), ('caa_num_epochs', 'num_epochs'), ('caa_save_path', 'save_path')]:
        v = getattr(args, arg_name)
        if v is not None:
            caa_tc[cfg_key] = v
            print(f'[main] overriding caa.train.{cfg_key} -> {v}')
    if args.coverage_min_gain is not None:
        config.setdefault('retrieval', {})['coverage_min_gain'] = args.coverage_min_gain
        print(f'[main] overriding retrieval.coverage_min_gain -> {args.coverage_min_gain}')
    datasets = parse_datasets(args.datasets)

    if args.mode == 'train':
        from src.train_stage import train_stage
        train_stage(curr_dir, datasets, args.model_name, args.augment_model, config, gold_only=args.gold_only, worker_id=args.worker_id, num_workers=args.num_workers, stage=args.stage, ranks_per_gpu=args.ranks_per_gpu)
    elif args.mode == 'test':
        from src.test_stage import test_stage
        if args.stage == 'offline':
            for ds_name, ds_type in datasets:
                test_stage(curr_dir, ds_name, ds_type, args.model_name, args.augment_model, config, k=args.k, silo_k=args.silo_k, gold_only=args.gold_only, num_silos=args.num_silos, worker_id=args.worker_id, num_workers=args.num_workers, stage=args.stage, caa_eval_tag=args.caa_eval_tag)
        else:
            assert len(datasets) == 1, f'test mode online/aggregate requires exactly one dataset:type pair (got {len(datasets)}: {datasets})'
            ds_name, ds_type = datasets[0]
            test_stage(curr_dir, ds_name, ds_type, args.model_name, args.augment_model, config, k=args.k, silo_k=args.silo_k, gold_only=args.gold_only, num_silos=args.num_silos, worker_id=args.worker_id, num_workers=args.num_workers, stage=args.stage, caa_eval_tag=args.caa_eval_tag)
