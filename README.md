# FedWeave: Federated Parametric RAG for Multi-Evidence Reasoning

Reference implementation for our paper **FedWeave: Federated Parametric RAG
for Multi-Evidence Reasoning** (ICDM 2026 submission).

FedWeave is a federated parametric RAG framework that answers queries
requiring complementary evidence distributed across silos, while raw
documents never leave their original silos. Each silo encodes its documents
into lightweight LoRA adapters; at inference, only adapters (not text) reach
the coordinator. The novelty is **evidence coverage as a shared signal across
retrieval and generation**: silos upload compact query-conditioned coverage
vectors that let the coordinator select complementary documents via
submodular set-level selection, and the same coverage signal then drives a
**Cross-Adapter Attention (CAA)** module that lets the selected adapters
exchange information in activation space before being aggregated into the
backbone LLM.

The codebase predates the final paper terminology and still names the CAA
module `CLA` (Cross-LoRA Attention) in source files and CLI flags — the
two refer to the same module.

| Headline (averaged across 8 QA subsets) | FedWeave | Strongest baseline | Δ |
|---|---|---|---|
| F1 (Table I in the paper) | **0.3836** | 0.3143 | **+22.1%** rel. |
| Win/loss/tie vs 10 baselines | 8 / 0 / 0 | — | — |
| 2WQA Bridge / Compare / Compose (multi-evidence) | 0.6036 / 0.6471 / 0.0941 | 0.5447 / 0.5730 / 0.0802 | **+10.8%** / +13.0% / +17.3% |
| Prefix-attack Target Prompts (lower = safer) | 12.53 | 44.00 | **−71.5%** |
| Prefix-attack Repeat Prompts (lower = safer) | 1.27 | 22.86 | **−94.4%** |

Full per-combo table, baseline parameters, micro-benchmark numbers, and
paper-ready prose live in [docs/v6_vs_fedmosaic.md](docs/v6_vs_fedmosaic.md).

---

## Method in 30 seconds

```
                  Silos {S_1, ..., S_N}                Coordinator C
                  ─────────────────────                ─────────────────
query q   ─►   local candidates D_i   ──┐
              + coverage vectors C_d   ─┴──►  set-level greedy select   (Sec IV)
                                              k complementary docs D
                                                       │
              upload LoRA adapters {A_d}  ◄────────────┘
                                                       ▼
                                      h_i = A_i · x_in        (per-adapter activation)
                                      α_ij ∝ softmax(QK/√d + log R_ij)
                                      h~_i = h_i + γ · W_O Σ α_ij V_j
                                      x_out = W_base x_in + Σ β_i h~_i        (Sec V)
                                                       ▼
                                                     answer a
```

Two pieces, one shared signal (the coverage vectors):

- **Coverage-Aware Federated Retrieval** ([src/silo.py](src/silo.py),
  [src/r_matrix.py](src/r_matrix.py)). Each silo runs ColBERT MaxSim
  locally; for each candidate `d`, the coverage vector
  `C_d[i] = max_e ⟨e_q^i, e⟩` records how strongly the document supports
  each query token (paper Eq. 3). The coordinator picks `k` documents
  greedily to maximize `U(D) = Σ_i max_{d∈D} C_d[i]` (Eq. 4–5) — a monotone
  submodular objective with `(1 − 1/e)` guarantee (Theorem 1).
- **Cross-Adapter Collaborative Generation** ([src/cla.py](src/cla.py)).
  Per-doc activations `h_i = A_i x_in` are kept separate; a coverage-derived
  complementarity matrix `R_ij = 1 − cos(C_i, C_j)` biases cross-adapter
  attention so each `h_i` attends more to adapters covering different facets
  (Eq. 8–11); a coverage-aware aggregator (Eq. 12–13) then folds the refined
  activations back into the residual stream.

---

## Repository layout

```
FedHop/
├── main.py                       # CLI dispatcher (--mode train/test, --stage offline/cla/online/aggregate)
├── config.yaml                   # All paths and hyperparameters
├── requirements.txt
├── src/                          # Implementation (10 files, ~2.2k LOC) — see src/README.md
│   ├── cla.py                    # Cross-Adapter Attention module + MountedLoRAs context manager (paper Sec V)
│   ├── cla_train.py              # CAA training loop (DDP-aware, paper Eq. 14)
│   ├── r_matrix.py               # ColBERT encoder + coverage vectors + complementarity matrix (paper Eq. 3, 8)
│   ├── silo.py                   # Per-silo retrieval + coverage-greedy server selection (paper Sec IV-B)
│   ├── test_stage.py             # Federated inference pipeline
│   ├── train_stage.py            # Centralized pre-training pipeline
│   ├── inference.py              # Single-question CAA inference helper
│   ├── lora.py                   # Per-doc LoRA training (paper Eq. 1)
│   ├── dataset.py                # BM25 retrieve + augment + Dirichlet silo split
│   └── utils.py                  # Prompts, eval, I/O
├── scripts/                      # Sweep launchers + utilities — see scripts/README.md
├── micro_benchmark/              # 4 ablations / analyses (paper Sec VI-C, VI-D) — see micro_benchmark/README.md
│   ├── retrieval/                # Coverage-greedy vs BM25/Dense/ColBERT + complementarity heatmap (Fig 5)
│   ├── cla/                      # CAA-module ablation (Fig 6)
│   ├── gamma/                    # Conditioning-strength γ robustness (Fig 7)
│   └── attack/FedWeave/          # Target + Prefix privacy attack (Table II)
├── baselines/                    # 10 baseline implementations from 4 categories
├── docs/                         # Canonical documentation — see docs/README.md
│   ├── v6_vs_fedmosaic.md        # Main results table + Section 5 micro-benchmarks
│   ├── EXPERIMENT_LOG.md         # design iterations leading to the final FedWeave
│   └── paper_section_iiic_vs_code.md
├── output/                       # Symlink farm: output/{method}/{ds}/{type}/eval.json
└── logs/                         # Raw stdout per sweep — see logs/README.md
```

Anything under `_archive/` (in any directory) is from earlier iterations,
kept for audit; nothing current references it.

---

## Baselines (10 methods, 4 categories — paper Sec VI-A)

| Category | Methods | Where |
|---|---|---|
| Local RAG | Standard RAG, CoTRAG, ReAct | [baselines/{stanRAG, cotRAG, react}](baselines/) |
| In-context Federated RAG | FRAG, MKPQA, RAGRoute | [baselines/{c_fedrag, mkpqa, ragroute}](baselines/) |
| Federated Fine-Tuning | FedIT, FLoRA | [baselines/{fedit, flora}](baselines/) |
| Parametric RAG | PRAG, FedMosaic | [baselines/{prag, FedMosaic-main}](baselines/) |

All baselines share the silo split (Dirichlet α=0.1, N=6 silos),
retrieval budget (k=5 per silo), and backbone (LLaMA3.2-1B-Instruct).
See [baselines/README.md](baselines/README.md) for the three entry-point
patterns (unified dispatcher / FedMosaic native / dragin standalone).

---

## Installation

Python 3.10.4 with CUDA-enabled PyTorch.

```bash
cd FedHop
conda create -n fedweave python=3.10.4
conda activate fedweave
pip install -r requirements.txt
```

Edit [config.yaml](config.yaml) to set local paths for:

| Field | Meaning |
|---|---|
| `prep_dataset.split.split_model_path` | Local `all-MiniLM-L6-v2` (used for the dense silo split) |
| `cla.colbert_path` | Local `colbertv2.0` (used for coverage vectors + complementarity matrix) |
| `train.save_dir` | Where doc-LoRAs, CAA checkpoints, and eval outputs are written |
| `cla.R_cache_dir` | Where ColBERT doc-embeddings are cached |

Base LLM paths (`llama3.2-1b-instruct`, `llama3-8b-instruct`) are resolved
in [src/utils.py](src/utils.py) — update them to your local checkpoints.

---

## End-to-end commands

### 1. Prepare datasets (one-time)

If you have the released `dataset.tar.gz`:

```bash
tar -xzvf dataset.tar.gz       # produces dataset/train/ and dataset/test/
```

Otherwise rebuild from raw QA datasets via BM25 → augment → Dirichlet split.
See [src/dataset.py](src/dataset.py) and
[retriever_elasticsearch/](retriever_elasticsearch/) (Elasticsearch + DPR
Wikipedia dump). Evaluation datasets:

- **HotpotQA (HQA)** — multi-evidence over Wikipedia
- **2WikiMultihopQA (2WQA)** — multi-evidence with annotated reasoning paths
- **ComplexWebQuestions (CWQ)** — complex compositional multi-evidence over a KB
- **PopQA (PQA)** — single-evidence long-tail factual queries
- **Enron Emails + WikiText** — used only for the privacy micro-benchmark

### 2. Train FedWeave (offline doc-LoRAs + CAA module)

```bash
# Prewarm ColBERT scores cache (race-free, ~5 min single GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/prewarm_v6_scores.py

# Train per-doc LoRAs over the combined train pool (parallel, ~40 min × 6 GPUs)
bash scripts/launch_offline.sh

# Train CAA module across 3 γ_train values in parallel (~50 min)
bash scripts/run_v6_train_sweep.sh
```

Outputs: `<train.save_dir>/cla/v6_train{0.1,0.5,1.0}/cla_best.pt`.

### 3. Federated test (inference)

```bash
# Main γ_train × γ_infer sweep on 6 HQA + 2WQA combos (~95 min on 6 GPUs)
GPU_POOL="0 1 2 3 4 5" MAX_PARALLEL=25 \
  bash scripts/run_v6_sweep.sh

# Extend to PopQA + CWQ + τ early-stop sweep on all 8 combos (~50 min)
bash scripts/run_v6_phase2.sh

# Build the final comparison table
python scripts/build_main_results_table.py
```

Single-cell shortcut (one (dataset, type, γ_infer)):

```bash
python main.py \
    --mode test --stage all \
    --datasets hotpotqa:bridge \
    --num_silos 6 --k 5 \
    --cla_save_path /data1/liangzhilin/fedhop/cla/v6_train0.1/cla_best.pt \
    --cla_alpha 0.05
```

### 4. Run baselines (for the comparison table)

All 10 baselines share the silo split, retrieval budget, and backbone with
FedWeave. Three entry-point patterns — see
[baselines/README.md](baselines/README.md) for the full per-method
invocation:

```bash
# Unified dispatcher (8 RAG-style baselines)
python baselines/main.py --baseline {stanRAG,cotRAG,mkpqa,c_fedrag,ragroute,fedit,flora,prag} \
    --dataset hotpotqa --type bridge --k 5

# FedMosaic (native entry, has its own main.py + offline/online stages)
bash scripts/run_fedmosaic.sh             # 6 HQA + 2WQA combos
bash scripts/run_fedmosaic_popqa.sh       # PopQA

# ReAct + dragin (standalone, own argparse)
cd baselines/dragin && python main.py -c configs/llama1b/<config>.json
```

### 5. Micro-benchmarks (paper Sec VI-C, VI-D)

```bash
# Retrieval coverage (BM25 / Dense / ColBERT / FedWeave)   — Fig 5
cd micro_benchmark/retrieval && python compute_recall.py && python plot_recall.py

# CAA-module ablation (FedWeave vs FedWeave w/o CLA)        — Fig 6
cd micro_benchmark/cla && GPU_POOL='0 2 3 7' bash run_ablations.sh && python aggregate.py && python plot_ablation_focused.py

# γ sensitivity (3 γ_train × 8 γ_infer)                     — Fig 7
cd micro_benchmark/gamma && python aggregate.py && python plot_gamma.py

# Privacy attack (Target + Prefix on PII-injected Wiki)     — Table II
bash scripts/run_fedweave_master.sh
```

See [micro_benchmark/README.md](micro_benchmark/README.md) for headline
numbers per experiment.

---

## CLI reference

Every run goes through [main.py](main.py); the flags below are the most
common overrides (all others sit in [config.yaml](config.yaml)).

| Flag | Use |
|---|---|
| `--mode {train, test}` | train: centralized pre-training of CAA. test: federated inference. |
| `--stage {all, offline, cla, online, aggregate}` | Sub-stage. `train`: `offline` (doc-LoRAs) / `cla` (CAA module) / `all`. `test`: `offline` (per-silo LoRAs) / `online` (sharded inference) / `aggregate` / `all`. |
| `--datasets X:Y[,X:Y...]` | Comma-separated dataset:type pairs. `train` accepts many; `test` takes exactly one. |
| `--cla_alpha FLOAT` | Inference-time γ override (the conditioning strength in paper Eq. 11). Defaults to `cla.alpha` in config.yaml. |
| `--cla_save_path PATH` | Train: where to save CAA. Test: which checkpoint to load. |
| `--coverage_min_gain FLOAT` | τ early-stop threshold on the marginal coverage gain in the greedy selection (paper Eq. 5); always keeps ≥1 doc. |

Supported values:

| Flag | Values |
|---|---|
| `--model_name` | `llama3.2-1b-instruct`, `llama3-8b-instruct` |
| `--augment_model` | `llama3.2-1b-instruct` (default) |
| `--datasets` (test, by `dataset:type`) | `hotpotqa:{bridge,comparison}`, `2wikimultihopqa:{bridge_comparison,comparison,inference,compositional}`, `popqa:total`, `complexwebquestions:total` |

---

## Where results live

| What | Where |
|---|---|
| Main eval JSONs (per method × combo) | [output/{method}/{ds}/{type}/eval.json](output/) (symlinks; canonical files under `train.save_dir` or `baselines/.../output/`) |
| Final comparison table + paper Sec V prose | [docs/v6_vs_fedmosaic.md](docs/v6_vs_fedmosaic.md) |
| Micro-benchmark figures (PDF + PNG) | `micro_benchmark/{exp}/figures/` |
| Raw sweep stdout | [logs/](logs/) — see [logs/README.md](logs/README.md) for the per-experiment index |

To refresh the `output/` symlink farm after a new sweep finishes:

```bash
python scripts/collect_results.py
```

---

## Citing

```bibtex
@inproceedings{fedweave2026,
  title={FedWeave: Federated Parametric RAG for Multi-Evidence Reasoning},
  author={...},
  booktitle={ICDM},
  year={2026}
}
```

## Acknowledgements

Retrieval pipeline builds on [PRAG](https://github.com/oneal2000/prag) and
the [DPR](https://github.com/facebookresearch/DPR) Wikipedia dump. We thank
the authors of all 10 baselines for releasing their code; see
[baselines/README.md](baselines/README.md) for per-method references.
