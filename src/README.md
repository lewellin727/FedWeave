# src/ — what each file is for

10 files, ~2.2k lines total. Imported via `from src.X import ...` from
[main.py](../main.py) and the micro-benchmarks.

| File | Lines | Role |
|---|---|---|
| [caa.py](caa.py) | 254 | **Cross-Adapter Attention module** (inference). `CAAMLP` defines the per-MLP cross-attention block + token router. `MountedLoRAs` context manager swaps in N per-doc LoRAs and the R-cache for one forward. The 4 env-var ablation toggles (`CAA_NO_CROSS / NO_ROUTER / NO_R / NO_C`) are read each forward at [L124-128](caa.py#L124) — see [micro_benchmark/caa/README.md](../micro_benchmark/caa/README.md). |
| [caa_train.py](caa_train.py) | 300 | **CAA training loop**. DDP-aware, sweeps over per-question (LoRA combo, doc combo) pairs, distillation-style loss against the gold-context teacher. Driven by `--mode train --stage caa`. |
| [r_matrix.py](r_matrix.py) | 214 | **ColBERT-based R matrix**. `ColBERTEncoder` (singleton-cached) + `compute_R_via_colbert(question, docs)` → (R[N,N], scores[N]). R is the off-diag min-max normalized MaxSim doc-doc complementarity matrix with diag=1; scores is the query-doc MaxSim. Used by both inference and training. |
| [silo.py](silo.py) | 178 | **Federated retrieval**. `retrieve_with_coverage` (per-silo ColBERT top-k pool + coverage vectors) + `coverage_greedy_select` (server-side submodular merge over query tokens). Canonical entry for FedWeave retrieval. |
| [test_stage.py](test_stage.py) | 238 | **Test-mode pipeline**. `_federated_inference` runs 6-silo retrieval → coverage-greedy → CAA-mounted generation per question. Sub-stages: `offline` (per-silo LoRAs), `online` (inference, sharded), `aggregate` (merge shards → eval.json). |
| [train_stage.py](train_stage.py) | 102 | **Train-mode pipeline**. Sub-stages: `offline` (parallel doc-LoRA training over the combined train pool), `caa` (single-process CAA module training, requires LoRAs to exist), `all`. |
| [inference.py](inference.py) | 24 | **Single-question inference helper**. `caa_inference(question, base_model, caa_module, lora_paths, doc_passages, ...)` — computes R + scores, mounts LoRAs via `MountedLoRAs`, generates. Used by `test_stage._federated_inference` and the FedWeave attack online script. |
| [lora.py](lora.py) | 143 | **Per-doc LoRA training utilities**. `train_one_lora(doc, base_model, tokenizer, train_cfg)` — trains a rank-2 LoRA on `[down_proj, gate_proj, up_proj]` for 3 epochs (per-doc augmentation prompts). |
| [dataset.py](dataset.py) | 268 | **Dataset preparation**. Download, BM25 retrieve, augment (LLM-generated paraphrases), Dirichlet silo split. Driven by `--mode prep_dataset` (entry not currently in main.py — historical, run from scripts when rebuilding data). |
| [utils.py](utils.py) | 349 | **Common utilities**. Prompt templates, `model_generate`, EM/F1 eval, save_dir resolution, JSON I/O helpers. |

## Module dependency graph

```
main.py
 ├── train_stage.py ──┐
 └── test_stage.py ──┐│
                     ├┴─→ inference.py ──→ caa.py ──→ r_matrix.py
                     │                  └─→ utils.py
                     ├─→ silo.py     ──→ r_matrix.py
                     ├─→ lora.py     ──→ utils.py
                     └─→ caa_train.py ──→ caa.py, r_matrix.py
```

## Configuration

All paths and hyperparameters live in [../config.yaml](../config.yaml). The
CLI in [main.py](../main.py) exposes overrides for the most-frequently-tuned
fields (`--caa_alpha`, `--caa_save_path`, `--caa_eval_tag`,
`--coverage_min_gain`, `--max_entries`). Other fields require editing
config.yaml directly.
