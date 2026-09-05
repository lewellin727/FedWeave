# Source code overview

`main.py` dispatches training and evaluation to the modules in this directory.

| File | Purpose |
|---|---|
| `train_stage.py` | Trains document LoRA adapters and the CAA module. |
| `test_stage.py` | Runs document-adapter preparation, federated retrieval, generation, and evaluation. |
| `silo.py` | Implements silo-local ColBERT retrieval and global coverage-aware selection. |
| `r_matrix.py` | Computes coverage-derived complementarity and relevance signals. |
| `caa.py` | Implements Cross-Adapter Attention and mounts document LoRAs for generation. |
| `caa_train.py` | Trains CAA while keeping the backbone and document LoRAs frozen. |
| `lora.py` | Trains and loads per-document LoRA adapters. |
| `inference.py` | Provides single-query CAA inference. |
| `dataset.py` | Defines the processed dataset schema and CAA sample construction. |
| `utils.py` | Provides model loading, prompting, evaluation, and sharding utilities. |

Model paths and hyperparameters are configured in `../config.yaml`.
