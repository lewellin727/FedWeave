"""CAA module training — DDP-aware, multi-rank per GPU supported.

Frozen: base LLM (bf16) + all per-doc LoRAs (loaded via mmap on each step).
Trainable: CAA cross-attn + router params (no gate; alpha is a config scalar).

Per-step flow:
  1. Pick one CAA sample {question, answer, doc-LoRA paths, R}.
  2. Build input_ids = tokenize("Question: q\\nAnswer: " + answer); labels mask
     the prompt span with -100 so loss is only on the answer.
  3. Forward through base LLM with MountedLoRAs(caa_module, paths, R) — the CAAMLP
     wrappers route through the cross-attention path at layers L_caa.
  4. Backward; accumulate over `grad_accum_steps`; AdamW step.

DDP semantics:
  - All ranks share one CAA module replica; gradients are NCCL all-reduced on
    every backward via DDP wrapping the base_model.
  - Each rank pulls a disjoint sample shard (round-robin by global index).
  - Rank-0 owns all I/O: print, trajectory.json, ckpt save.
  - Val loss is computed by each rank on its shard, then summed via all_reduce.
"""
import os
import json
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.caa import CAAModule, mount_caa, unmount_caa, MountedLoRAs
from src.dataset import iter_caa_samples, get_doc_lora_paths, get_R_for_sample, assert_lora_paths_exist
from src.r_matrix import ColBERTEncoder
from src.utils import get_model


# ============================================================================
# DDP helpers
# ============================================================================

def is_ddp():
    return 'LOCAL_RANK' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1


def get_rank():
    return int(os.environ.get('RANK', 0))


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))


def is_main_rank():
    return get_rank() == 0


def setup_ddp(ranks_per_gpu=1):
    """Init NCCL + bind this rank to its physical GPU.

    Caller is expected to set CUDA_VISIBLE_DEVICES to the GPUs to use; then
    LOCAL_RANK / ranks_per_gpu gives the visible-GPU index for this rank.
    Returns (visible_gpu_idx, world_size, rank, is_main).
    """
    if not is_ddp():
        return None, 1, 0, True
    dist.init_process_group(backend='nccl')
    local_rank = get_local_rank()
    gpu_idx = local_rank // ranks_per_gpu
    torch.cuda.set_device(gpu_idx)
    return gpu_idx, get_world_size(), get_rank(), is_main_rank()


def cleanup_ddp():
    if is_ddp() and dist.is_initialized():
        dist.destroy_process_group()


def main_only_print(*args, **kwargs):
    if is_main_rank():
        print(*args, **kwargs)


# ============================================================================
# Helpers
# ============================================================================

def freeze_base_model(model):
    for p in model.parameters():
        p.requires_grad = False


def build_input_with_labels(question, answer, tokenizer, device):
    """Tokenize 'Question: q\\nAnswer: a' and mask the prompt span in labels (-100)."""
    prompt = f"Question: {question}\nAnswer: "
    full = prompt + answer
    enc = tokenizer(full, return_tensors='pt')
    prompt_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids
    prompt_len = prompt_ids.shape[1]
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return input_ids, attention_mask, labels


def assert_only_caa_trainable(base_model, caa_module):
    base_trainable = [n for n, p in base_model.named_parameters() if p.requires_grad and 'caa_module' not in n]
    if base_trainable:
        raise RuntimeError(f'unexpected trainable base params after freeze: {base_trainable[:5]}...')
    caa_trainable = sum(p.numel() for p in caa_module.parameters() if p.requires_grad)
    if caa_trainable == 0:
        raise RuntimeError('CAA module has 0 trainable params')
    return caa_trainable


def shard_samples(samples, rank, world):
    """Round-robin shard so the union across ranks reproduces the original order exactly once."""
    return [s for i, s in enumerate(samples) if i % world == rank]


# ============================================================================
# Single forward+backward
# ============================================================================

def caa_train_step(model_to_call, caa_module, sample, lora_paths, R, scores, tokenizer, optimizer, device, base_dtype, accum_steps=1, do_step=True):
    """One forward/backward pass (CAA module only is trainable). Returns task loss."""
    input_ids, attention_mask, labels = build_input_with_labels(sample['question'], sample['answer'], tokenizer, device)
    R_d = R.to(device) if torch.is_tensor(R) else torch.tensor(R, device=device, dtype=torch.float32)
    scores_d = scores.to(device) if torch.is_tensor(scores) else torch.tensor(scores, device=device, dtype=torch.float32)
    with MountedLoRAs(caa_module, lora_paths, R_d, scores_d, device=device, dtype=base_dtype):
        outputs = model_to_call(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        task_loss = outputs.loss
    (task_loss / accum_steps).backward()
    if do_step:
        optimizer.step()
        optimizer.zero_grad()
    return task_loss.item()


@torch.no_grad()
def caa_eval_loss(model_to_call, caa_module, samples, encoder, caa_config, tokenizer, device, base_dtype, save_dir, model_name, lora_epoch, world=1):
    """Mean loss over a held-out sample list. Under DDP, samples is the per-rank shard;
    returns the globally averaged loss via all_reduce.
    """
    was_training = model_to_call.training
    model_to_call.eval()
    caa_module.eval()
    total, n = 0.0, 0
    for s in samples:
        lora_paths = get_doc_lora_paths(s, save_dir, model_name, lora_epoch, mode='train')
        try:
            assert_lora_paths_exist(s, lora_paths)
        except FileNotFoundError:
            continue
        R, scores = get_R_for_sample(s, encoder, caa_config, mode='train')
        R = R.to(device); scores = scores.to(device)
        input_ids, attention_mask, labels = build_input_with_labels(s['question'], s['answer'], tokenizer, device)
        with MountedLoRAs(caa_module, lora_paths, R, scores, device=device, dtype=base_dtype):
            out = model_to_call(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total += out.loss.item()
        n += 1
    if world > 1:
        t = torch.tensor([total, float(n)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, n = float(t[0].item()), int(t[1].item())
    if was_training:
        model_to_call.train()
    caa_module.train()
    return total / max(n, 1), n


# ============================================================================
# Full training loop
# ============================================================================

def caa_train(combos, model_name, augment_model, config, num_epochs=3, lr=1e-4, grad_accum_steps=8, weight_decay=0.01, log_every=10, val_every=50, val_ratio=0.1, k_per_combo=50, seed=42, save_path=None, dtype=torch.bfloat16, ranks_per_gpu=1):
    """Train CAAModule on samples from `combos`. DDP-aware.

    When LOCAL_RANK / WORLD_SIZE env vars are set (torchrun), runs in DDP mode:
      - each rank binds to GPU index = LOCAL_RANK // ranks_per_gpu
      - samples are round-robin sharded across ranks
      - rank-0 owns all printing, trajectory.json, and ckpt save
    Single-process invocation works transparently (no env vars => world=1).
    """
    gpu_idx, world, rank, is_main = setup_ddp(ranks_per_gpu=ranks_per_gpu)
    save_dir = config['train']['save_dir']
    lora_epoch = config['train']['lora_epoch']
    caa_config = config['caa']

    device_map = {"": f"cuda:{gpu_idx}"} if gpu_idx is not None else "auto"
    base_model, tokenizer, _ = get_model(model_name, config, dtype=dtype, device_map=device_map)
    base_model.train()
    freeze_base_model(base_model)
    base_dtype = next(base_model.parameters()).dtype
    device = next(base_model.parameters()).device
    main_only_print(f'=== Base model {model_name} dtype={base_dtype} world={world} rank={rank} device={device} ===')

    alpha = float(caa_config.get('alpha', 0.1))
    caa_module = CAAModule(hidden_size=base_model.config.hidden_size, intermediate_size=base_model.config.intermediate_size, L_caa=caa_config['L_caa'], d_a=caa_config['d_a'], alpha=alpha).to(device)
    main_only_print(f'=== alpha = {alpha:.3g} ===')

    mount_caa(base_model, caa_module, caa_config['L_caa'])
    n_trainable = assert_only_caa_trainable(base_model, caa_module)
    main_only_print(f'=== CAA mounted on layers {caa_config["L_caa"]}, trainable params: {n_trainable:,} ===')

    if world > 1:
        model_to_call = DDP(base_model, device_ids=[gpu_idx], find_unused_parameters=False, gradient_as_bucket_view=True)
        main_only_print(f'=== DDP wrap done (find_unused=False, world={world}) ===')
    else:
        model_to_call = base_model

    encoder = ColBERTEncoder(caa_config['colbert_path'], device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(caa_module.parameters(), lr=lr, weight_decay=weight_decay)
    main_only_print(f'=== Optimizer: AdamW lr={lr} wd={weight_decay} grad_accum={grad_accum_steps} epochs={num_epochs} ===')

    all_samples = list(iter_caa_samples(combos, augment_model, k_per_combo=k_per_combo, mode='train', seed=seed))
    rng = torch.Generator().manual_seed(seed)
    val_n = int(round(val_ratio * len(all_samples)))
    val_idx_set = set(torch.randperm(len(all_samples), generator=rng).tolist()[:val_n])
    val_samples_global = [all_samples[i] for i in sorted(val_idx_set)]
    train_samples_global = [s for i, s in enumerate(all_samples) if i not in val_idx_set]
    train_samples = shard_samples(train_samples_global, rank, world)
    val_samples = shard_samples(val_samples_global, rank, world)
    main_only_print(f'=== Split: train_global={len(train_samples_global)} val_global={len(val_samples_global)} | this rank: train={len(train_samples)} val={len(val_samples)} ===')

    trajectory = {'config': {'num_epochs': num_epochs, 'lr': lr, 'grad_accum_steps': grad_accum_steps, 'val_ratio': val_ratio, 'val_every': val_every, 'log_every': log_every, 'n_train': len(train_samples_global), 'n_val': len(val_samples_global), 'dtype': str(dtype), 'L_caa': caa_config['L_caa'], 'alpha': alpha, 'world_size': world, 'ranks_per_gpu': ranks_per_gpu}, 'steps': []}
    save_dir_for_ckpt = os.path.dirname(save_path) if save_path else None
    if save_dir_for_ckpt and is_main:
        os.makedirs(save_dir_for_ckpt, exist_ok=True)
    trajectory_path = os.path.join(save_dir_for_ckpt, 'trajectory.json') if save_dir_for_ckpt else None
    best_save_path = save_path.replace('.pt', '_best.pt') if save_path else None

    optimizer.zero_grad()
    accum_count = 0
    step = 0
    losses_window = []
    best_val = float('inf')
    best_step = -1
    t0 = time.time()

    for epoch in range(num_epochs):
        epoch_rng = torch.Generator().manual_seed(seed + epoch * 1000 + rank)
        order = torch.randperm(len(train_samples), generator=epoch_rng).tolist()
        for s_idx in order:
            sample = train_samples[s_idx]
            lora_paths = get_doc_lora_paths(sample, save_dir, model_name, lora_epoch, mode='train')
            try:
                assert_lora_paths_exist(sample, lora_paths)
            except FileNotFoundError:
                continue
            R, scores = get_R_for_sample(sample, encoder, caa_config, mode='train')
            accum_count += 1
            do_step = (accum_count == grad_accum_steps)
            loss_val = caa_train_step(model_to_call, caa_module, sample, lora_paths, R, scores, tokenizer, optimizer, device, base_dtype, accum_steps=grad_accum_steps, do_step=do_step)
            losses_window.append(loss_val)
            if do_step:
                accum_count = 0
                step += 1
                if step % log_every == 0 and is_main:
                    avg_loss = sum(losses_window) / len(losses_window)
                    losses_window = []
                    elapsed = time.time() - t0
                    record = {'step': step, 'epoch': epoch, 'train_loss': avg_loss, 'elapsed_sec': elapsed}
                    print(f'[e{epoch} s{step:5d}] loss={avg_loss:.4f}  elapsed={elapsed:.0f}s')
                    if step % val_every == 0:
                        val_loss, n_used = caa_eval_loss(model_to_call, caa_module, val_samples, encoder, caa_config, tokenizer, device, base_dtype, save_dir, model_name, lora_epoch, world=world)
                        record['val_loss'] = val_loss
                        record['val_n_used'] = n_used
                        marker = ''
                        if val_loss < best_val:
                            best_val = val_loss
                            best_step = step
                            if best_save_path:
                                torch.save(caa_module.state_dict(), best_save_path)
                                marker = '  ↓ NEW BEST (saved)'
                        print(f'    val_loss={val_loss:.4f} (n={n_used}){marker}')
                    trajectory['steps'].append(record)
                    with open(trajectory_path, 'w') as f:
                        json.dump(trajectory, f, indent=2)
                elif step % val_every == 0 and world > 1:
                    caa_eval_loss(model_to_call, caa_module, val_samples, encoder, caa_config, tokenizer, device, base_dtype, save_dir, model_name, lora_epoch, world=world)

    if is_main and save_path:
        torch.save(caa_module.state_dict(), save_path)
        print(f'\nSaved final CAA ckpt -> {save_path}')
        if best_step >= 0:
            print(f'Best-val CAA ckpt    -> {best_save_path}  (val_loss={best_val:.4f} @ step {best_step})')
        if trajectory_path:
            trajectory['best_val_loss'] = best_val
            trajectory['best_step'] = best_step
            trajectory['total_elapsed_sec'] = time.time() - t0
            with open(trajectory_path, 'w') as f:
                json.dump(trajectory, f, indent=2)
            print(f'Saved trajectory     -> {trajectory_path}')

    unmount_caa(base_model)
    cleanup_ddp()
    return caa_module
