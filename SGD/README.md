# SGD Experiments

This repo runs controlled classification experiments on synthetic string/number datasets (e.g., **50‑digit decimal** or **N‑bit binary**), where the ground‑truth labels come from **explicit mathematical/logical rules** (parity, divisibility, primality, Dyck‑2, pattern triggers, etc.).

> If you only need a quick run: jump to **Quickstart**.


---

## Repository structure (high‑level)

```
collators.py         # Tokenizer-aware batch collation (LLM inputs) and label formatting for BCEWithLogitsLoss
dataloaders.py       # Dataset factories and PyTorch DataLoader builders
main.py              # Single-run training + evaluation entrypoint
models.py            # Model definitions/wrappers (e.g., Qwen/Llama/DeepSeek variants) + binary classifier heads
sweep.py             # Hyperparameter sweep orchestrator (grid/variants), creates subfolders & logs per run
target_functions.py  # Ground-truth labeling functions (parity, divisibility, prime, Dyck-2, patterns, etc.)
```

---

## Dependencies

A complete list of dependencies with exact versions is provided in the `environment.yml` file.

---

## Setup

1. **Create a directory called `results_new`:**
    ```bash
    mkdir results_new
    ```

2. **Create a dummy directory called `0` inside the `results_new` directory:**
    ```bash
    mkdir results_new/0
    ```

## Experiments

### Sweep - Quickstart

### Running the Code

To run the code, use the following command:
```bash
python sweep.py
```

### Running Example

To initiate five runs with {lr=0.1 and batch size 5} and five runs with {lr=0.2 and batch size 5}:
```bash

$ python sweep.py
Please specify filepath of base settings:
./conf/base_settings.py

Specify hyperparam to vary (e.g. lr) or 'END':
lr

Specify value of hyperparam and how many as AxB or 'END':
0.1x5
0.2x5
END

Specify hyperparam to vary (e.g. lr) or 'END':
batch_size

Specify value of hyperparam and how many as AxB or 'END':
5x1
END

Specify hyperparam to vary (e.g. lr) or 'END':
END

You are about to queue 20 jobs. Would you like to proceed? (y/n)
y
```

### Example Sweep qwen3 1.7B experiments (binary) (same values from paper)
```bash
$ python sweep.py
Please specify filepath of base settings:
./conf/base_settings.py

Specify hyperparam to vary (e.g. lr) or 'END':
  train_set_size
    Specify value of hyperparam and how many as AxB or 'END':
    200x1
    END

Specify hyperparam to vary (e.g. lr) or 'END':
  test_set_size
    Specify value of hyperparam and how many as AxB or 'END':
    10000x1
    END

Specify hyperparam to vary (e.g. lr) or 'END':
  batch_size
    Specify value of hyperparam and how many as AxB or 'END':
    20x1
    END

Specify hyperparam to vary (e.g. lr) or 'END':
  lr
    Specify value of hyperparam and how many as AxB or 'END':
    1e-5x1
    END

Specify hyperparam to vary (e.g. lr) or 'END':
  eta_min
    Specify value of hyperparam and how many as AxB or 'END':
    1e-6x1
    END

Specify hyperparam to vary (e.g. lr) or 'END':
  sequence_length
    Specify value of hyperparam and how many as AxB or 'END':
    100x1
    50x1
    30x1
    25x1
    20x1
    END

Specify hyperparam to vary (e.g. lr) or 'END':
  model
    Specify value of hyperparam and how many as AxB or 'END':
    ’qwen1.7B’x1
    END

Specify hyperparam to vary (e.g. lr) or 'END':
  target_func
    Specify value of hyperparam and how many as AxB or 'END':
    ’func7’x1 
    ’func1’x1
    ’func20’x1
    ’func21’x1
    ’func15’x1
    ’func18’x1
    ’func17’x1
    ’func3’x1
    'func4'x1
    END 

Specify hyperparam to vary (e.g. lr) or 'END':
  n_epochs
    Specify value of hyperparam and how many as AxB or 'END':
    200x1
    END 
```

## Now we will list some experiment configs. They can be run using the above process (or scripts, ...).


## Sweep — qwen3 1.7B (decimal)

| Field | Value / Options | Notes |
|---|---|---|
| **Models** | `qwen1.7B` | |
| **Target Functions** | `func19` `func22` | 2 total |
| **Sequence Lengths** | `100`, `50`, `30`, `25`, `20` | 2 total |
| **Train Set Size** | `200` | |
| **Test Set Size** | `10000` | |
| **Batch Size** | `20` | |
| **Epochs** | `200` | |
| **LR** | `1e-5` | |
| **Eta Min** | `1e-6` | |
| **BOS Token** | `10`| |
| **Vocab Size** | `11` | |
| **Expected Runs** | **10** | 5 seq lengths × 2 functions |

---

## Sweep — Training Model ablation from scratch

| Field | Value / Options | Notes |
|---|---|---|
| **Models** | `qwen1.7B` `llama3` `deepseek` | 3 total |
| **Target Functions** | `func18` `func21` `func17` | 3 total |
| **Sequence Lengths** | `100`, `50`, `30`, `25`, `20` | 5 total |
| **Train Set Size** | `200` | |
| **Test Set Size** | `10000` | |
| **Batch Size** | `20` | |
| **Epochs** | `200` | |
| **LR** | `1e-5` | |
| **Eta Min** | `1e-6` | |
| **Expected Runs** | **45** | 5 seq lengths × 3 functions × 3 Models |

---

## Sweep — Training Model from pretrained models

| Field | Value / Options | Notes |
|---|---|---|
| **Models** | `qwen3_finetune` | |
| **Target Functions** | `func18` `func3` `func22` | 3 total |
| **Sequence Lengths** | `100`, `50`, `30`, `25`, `20` | 5 total |
| **Train Set Size** | `200` | |
| **Test Set Size** | `10000` | |
| **Batch Size** | `20` | |
| **Epochs** | `1000` | |
| **LR** | `5e-3` | |
| **Eta Min** | `1e-3` | |
| **Num Layers to Finetune** | `8` `4` `2` | 3 total |
| **Expected Runs** | **45** | 5 seq lengths × 3 functions × 3 layers |

Note: Here 0 will mean finetuning on full model.

---

## Sweep — Bloom training on large dataset

| Field | Value / Options | Notes |
|---|---|---|
| **Models** | `bloom` | 3 total |
| **Target Functions** | `func18` `func3` `func22` | 3 total |
| **Sequence Lengths** | `100`, `50`, `30`, `25`, `20` | 5 total |
| **Train Set Size** | `100000` | |
| **Test Set Size** | `10000` | |
| **Batch Size** | `256` | |
| **Epochs** | `1000` | |
| **LR** | `1e-5` | |
| **Eta Min** | `1e-5` | |
| **n_heads** | `8` | |
| **n_embd** | `512` | |
| **n_layers** | `24` | |
| **BOS Token** | `10`| For decimal only |
| **Vocab Size** | `11` | For decimal only |
| **Expected Runs** | **15** | 5 seq lengths × 3 functions  |


---

## Similarly use ablations all other ablations can be performed from the config defined in paper.


## Results & logging

A common structure looks like:
```
results_new/
└── 260/                         # experiment_id (auto-increment)
    ├── 0/                       # model_id or attempt_id
    │   ├── config.txt
    │   ├── logs.log
    │   └── metrics.py
    └── 1/
        └── ...
```
> Exact filenames can differ; the key is: **each attempt lives in its own subfolder** with self‑describing artifacts.


---

## Reproducibility

- Always set an explicit **seed** (e.g., `--seed 42`).
- Log the full config (`config.json`) for every run.
- Persist the **train/test split** per `(target_function, seq_len)`.

---