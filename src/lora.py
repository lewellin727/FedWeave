"""Doc-LoRA training and loading primitives.

Everything that deals with a single per-document LoRA adapter on disk:
  - `train_lora`:              train one adapter and save it bf16 to disk.
  - `load_doc_lora_weights`:   load a saved PEFT adapter into a per-layer tensor dict
                               (used by CLA's MountedLoRAs to attach LoRAs at runtime).

Adapter directory layout matches PEFT's `save_pretrained`:
  <lora_dir>/<doc_id>/adapter_config.json
                      adapter_model.safetensors

All saves are bf16 (configured in `train_lora`) to halve disk usage; load reads
whatever dtype is on disk and the caller (e.g. MountedLoRAs) re-casts for forward.
"""
import os
import re
import gc
import json

import torch
from safetensors.torch import load_file
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from transformers import TrainingArguments


LORA_KEY_RE = re.compile(r'.*\.layers\.(\d+)\.mlp\.(\w+)_proj\.lora_(A|B)\.weight$')


# ============================================================================
# Training: build one bf16 LoRA adapter for a single Document
# ============================================================================

def train_lora(doc, doc_id, base_model, tokenizer, lora_dir, config):
    """Train one LoRA adapter for a single Document and save it in bf16."""
    lora_path = os.path.join(lora_dir, doc_id)
    if os.path.exists(lora_path):
        print(f'lora {doc_id} exists')
        return

    train_data = doc.get_train_data(tokenizer)
    if len(train_data) == 0:
        print(f'lora {doc_id}: SKIPPED — empty train data (augments missing or empty)')
        return
    lora_config = LoraConfig(r=config['train']['lora_rank'], lora_alpha=config['train']['lora_alpha'], target_modules=config['train']['target_modules'], lora_dropout=0, task_type=TaskType.CAUSAL_LM, inference_mode=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(config['train']['training_output_dir'], 'lora', doc_id),
        save_total_limit=0,
        per_device_train_batch_size=config['train']['batch_size'],
        gradient_accumulation_steps=1,
        learning_rate=float(config['train']['lr']),
        num_train_epochs=config['train']['lora_epoch'],
        save_strategy="no",
        bf16=True,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
    )

    model = get_peft_model(base_model, lora_config)

    trainer = SFTTrainer(model=model, train_dataset=train_data, args=training_args)
    trainer.train()

    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)

    model.save_pretrained(lora_path, safe_serialization=True)
    print(f'Saved bf16 lora at {lora_path}')

    model = model.unload()
    if isinstance(model, PeftModel):
        model = model.base_model.model
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# Loading: parse PEFT-saved adapter into a per-layer (A, B) tensor dict
# ============================================================================

def load_doc_lora_weights(lora_path):
    """Load one PEFT adapter folder into a structured dict (CPU tensors, source dtype).

    Returns:
        {
            'scaling': float,            # lora_alpha / r (PEFT default)
            'by_layer': {
                layer_idx: {'gate': (A, B), 'up': (A, B), 'down': (A, B)},
                ...
            }
        }
    Each A has shape (r, in_features); each B has shape (out_features, r) — matching
    PEFT's nn.Linear convention so forward is `(x @ A.T) @ B.T * scaling`.
    """
    cfg = json.load(open(os.path.join(lora_path, 'adapter_config.json')))
    scaling = float(cfg['lora_alpha']) / float(cfg['r'])
    weights = load_file(os.path.join(lora_path, 'adapter_model.safetensors'))

    by_layer = {}
    for key, tensor in weights.items():
        m = LORA_KEY_RE.match(key)
        if not m:
            continue
        l, proj, ab = int(m.group(1)), m.group(2), m.group(3)
        by_layer.setdefault(l, {}).setdefault(proj, {})[ab] = tensor

    result = {'scaling': scaling, 'by_layer': {}}
    for l, projs in by_layer.items():
        layer_dict = {}
        for p in ('gate', 'up', 'down'):
            if p in projs and 'A' in projs[p] and 'B' in projs[p]:
                layer_dict[p] = (projs[p]['A'], projs[p]['B'])
        if layer_dict:
            result['by_layer'][l] = layer_dict
    return result


# ============================================================================
# Device shipping (no persistent cache — safetensors uses mmap so warm reads
# are served from the OS page cache, shared across DDP ranks for free)
# ============================================================================

def load_doc_lora_on_device(lora_path, device, dtype):
    """load_doc_lora_weights() + ship tensors to (device, dtype) for one step.

    No persistent cache: each call re-reads via safetensors' mmap; the OS page
    cache makes warm reads ~0.1ms each. Allocations on `device` are reclaimed by
    PyTorch's caching allocator when the returned dict falls out of scope, so
    GPU memory stays bounded regardless of corpus size (critical when scaling
    to 50K+ LoRAs where holding all of them on GPU would OOM).
    """
    import torch
    adapter = load_doc_lora_weights(lora_path)
    for l_data in adapter['by_layer'].values():
        for p in ('gate', 'up', 'down'):
            if p in l_data:
                a, b = l_data[p]
                l_data[p] = (a.to(device, dtype=dtype, non_blocking=True), b.to(device, dtype=dtype, non_blocking=True))
    return adapter
