# FedWeave: Federated Parametric RAG for Multi-Evidence Reasoning

Reference implementation of *FedWeave* — a federated parametric RAG framework
for queries whose answers depend on complementary evidence scattered across
silos, while raw documents stay local. Both retrieval and generation share a
single signal: **evidence coverage**.

- **Coverage-aware retrieval (paper §IV).** Each silo summarizes how each
  candidate document supports each query token as a query-conditioned
  *coverage vector* (ColBERT late-interaction). Only the vectors + doc IDs
  leave the silo; the coordinator then runs a greedy submodular selection that
  maximizes set-level coverage — within (1 − 1/e) of optimal.
- **Cross-Adapter Collaborative Generation (paper §V).** Selected document
  LoRA adapters are kept separate in activation space at the coordinator. A
  **Cross-Adapter Attention (CAA)** module reuses the coverage vectors as a
  complementarity prior, lets adapters exchange information, then aggregates
  them into the frozen backbone LLM.

> Code uses the legacy name `CLA` (Cross-LoRA Attention) for what the paper
> calls `CAA`. Same module.

## Setup

```bash
pip install -r requirements.txt           # Python 3.10+
# then edit config.yaml: backbone LLM, ColBERTv2, dataset root
```

## Quick start

`main.py` is the single entry point. The pipeline has an offline phase
(per-doc LoRAs + CAA training) and an online phase (federated retrieval +
collaborative generation).

```bash
# Offline 1: per-document LoRA adapters (once per dataset/type)
python main.py --mode test --stage offline \
    --datasets hotpotqa:bridge --num_silos 6

# Offline 2: train the CAA module at the coordinator
python main.py --mode train --stage cla \
    --datasets hotpotqa:bridge,hotpotqa:comparison,2wikimultihopqa:comparison \
    --cla_save_path /path/to/cla.pt

# Online: federated retrieval + collaborative generation
python main.py --mode test --stage all \
    --datasets 2wikimultihopqa:comparison \
    --num_silos 6 --k 5 \
    --cla_save_path /path/to/cla.pt --cla_alpha 0.05
```

Useful flags: `--silo_k` (per-silo top-k, defaults to `--k`),
`--coverage_min_gain` (greedy early-stop threshold τ),
`--cla_eval_tag` (eval subdir).

## Datasets

Four open-domain QA benchmarks: **HotpotQA** (bridge, comparison),
**2WikiMultihopQA** (bridge_comparison, comparison, inference, compositional),
**ComplexWebQuestions**, **PopQA**. Each is partitioned into 6 silos under a
Dirichlet allocation (α = 0.1); default retrieval `k = 5`.

## Code organization

The `src/` files map to the paper as follows:

| Paper section | Files |
|---|---|
| §III workflow (Algorithm 1) | [main.py](main.py), [src/test_stage.py](src/test_stage.py), [src/train_stage.py](src/train_stage.py) |
| §IV Coverage-aware retrieval | [src/r_matrix.py](src/r_matrix.py) (coverage vector, Eq. 3) · [src/silo.py](src/silo.py) (local search + greedy selection, Eq. 5) |
| §V Cross-adapter generation | [src/cla.py](src/cla.py) (CAA, Eq. 6–13 + Alg. 2) · [src/cla_train.py](src/cla_train.py) (training, Eq. 14) · [src/inference.py](src/inference.py) (online generation) |
| Doc-LoRA training (Eq. 1) | [src/lora.py](src/lora.py), [src/train_stage.py](src/train_stage.py) |
| Data + prompts | [src/dataset.py](src/dataset.py), [src/utils.py](src/utils.py) |

## Citation

Anonymous submission; citation will be added after acceptance.
