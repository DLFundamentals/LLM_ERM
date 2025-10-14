# LTC Function-Synthesis Runner — README

This repo contains a experiment that prompts an OpenAI GPT-5 model to **synthesize a concise Python function** `f(x)` that matches a hidden target mapping. Datasets are generated from ground-truth functions (binary/decimal, parity/automata/prime/etc.), split deterministically, and persisted to disk. Each model attempt returns code, which is compiled and evaluated; results are logged and exported to JSONL/CSV.

---

## Repository layout

```
.
├── runner.py                  # Main orchestrator (async requests, eval, logging, artifacts)
├── data_handler.py        # Dataset generators (binary/decimal/prime/Dyck/etc.)
├── target_functions.py    # Ground-truth functions and TARGET_FUNCTIONS registry
└── runner.log                 # (created at runtime) JSON logs
```

---

## Dependencies

A complete list of dependencies with exact versions is provided in the `environment.yml` file.

---

## Quick start (For a single function - Full Parity)

1) **Set your API key**:

```bash
export OPENAI_API_KEY=sk-...
```

2) **Do Not Run default grid if resources are limited(Runs all task). Instead run the following command.** 

```bash
python runner.py   --functions fn_a   --lengths 100 50 30 25 20 --enable-code-interpreter
```

3) **Artifacts** (written at the end):

- `results_attempts.jsonl` — all attempts with raw text and usage payloads
- `results_attempts.csv` — curated, analysis-friendly columns
- `datasets/` — persisted train/val/test under target/length/seed
- `runner.log` — structured JSON logs for every step

---

---

## Replicate entire expriments

1) **Set your API key**:

```bash
export OPENAI_API_KEY=sk-...
```

2) **Run this command to replicate. Note : This is long task, will cost $. It will run all funtions (fn_a fn_b fn_c fn_d fn_e fn_f fn_g fn_h fn_i fn_j fn_k ) for all the dimenstions (100 50 30 25 20). All other config used in paper are set as default.** 

```bash
python runner.py --enable-code-interpreter
```

3) **Artifacts** (written at the end):

- `results_attempts.jsonl` — all attempts with raw text and usage payloads
- `results_attempts.csv` — curated, analysis-friendly columns
- `datasets/` — persisted train/val/test under target/length/seed
- `runner.log` — structured JSON logs for every step

---

## CLI usage

```bash
python runner.py   --functions fn_a fn_j fn_i   --lengths 100 50   --attempts 8   --model gpt-5   --max-output-tokens 20000   --concurrency 5   --timeout 1200   --enable-code-interpreter   --verbosity low   --reasoning-effort high   --train-size 100 --val-size 100 --test-size 10000   --seed 42   --dataset-dir datasets   --out-jsonl results_attempts.jsonl   --out-csv results_attempts.csv
```

### Environment variables (defaults)

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL=gpt-5`
- `MAX_OUTPUT_TOKENS=20000`
- `REASONING_EFFORT=high` (`minimal|medium|high`)
- `TEXT_VERBOSITY=low` (`low|medium|high`)
- `TOOL_CHOICE=auto` (`auto|none`)
- `ENABLE_CODE_INTERPRETER=0` (`1` to enable)
- `CONCURRENCY=5`
- `PER_CALL_TIMEOUT_S=1200`
- `ATTEMPTS=5`
- `TRAIN_SIZE=100`, `VAL_SIZE=100`, `TEST_SIZE=10000`
- `GLOBAL_SEED=42`
- `DATASET_DIR=datasets`
- `OUT_JSONL=results_attempts.jsonl`, `OUT_CSV=results_attempts.csv`
- `LOG_LEVEL=INFO`
- `DRY_RUN=0`

> **Dry run** (`--dry-run` or `DRY_RUN=1`) prints the exact constructed prompt for each query and **does not** call the API.

---

## What the runner actually does

1) **Derive deterministic seed** per `(fn, L)`:
   ```
   derived_seed = (hash((fn, L)) & 0x7fffffff) ^ global_seed
   ```
   Datasets are saved under:
   ```
   datasets/<target_name>/L<length>/seed<derived_seed>/{train.txt,val.txt,test.txt,meta.json}
   ```

2) **Generate or reuse data**  
   If files exist with exact sizes, they’re reused; else they’re regenerated with the derived seed.

3) **Build prompt**  
   Uses a small batch of training examples, with a **binary** or **decimal** problem statement depending on the target (see `DECIMAL_FNS`). The user content ends with:
   ```
   You must output ONLY a single JSON object: {"code": "<python function>"}.
   ```

4) **Call Responses API** (async, concurrency-limited)  
   Code Interpreter tool injection via `--enable-code-interpreter`. One retry for transient errors. **MUST use this to replicate the results**

5) **Extract & compile code**  
   - Parse output as JSON; fallback to `{ ... }` regex if needed.  
   - Remove Markdown fences, dedent, parse AST.  
   - Prefer `def f(x): ...`; otherwise take the first function.  
   - `exec` in a restricted global namespace (`__builtins__` only) and retrieve the callable.

6) **Evaluate**  
   - Validate on `val.txt` via `_local_get_accuracy` (`external_get_accuracy` hook supported).  
   - If `val_acc == 1.0`, evaluate on test and **early-stop**.  
   - Otherwise continue attempts up to `--attempts`.

7) **Log & export**  
   Every step emits a JSON record to stdout + `runner.log`. Final arrays of rows are written to JSONL/CSV.

---

## Targets and functions

`runner.py` uses a compact “experiment ID” → target mapping:

| ID   | `FUNCTION_NAME_MAPPING` target | Domain |
|------|-------------------------------|--------|
| fn_a | `parity_all`                  | binary |
| fn_b | `parity_first_half`           | binary |
| fn_c | `patternmatch1`               | binary |
| fn_d | `patternmatch2`               | binary |
| fn_e | `parity_rand_3`               | binary |
| fn_f | `parity_rand_10`              | binary |
| fn_g | `palindrome`                  | binary |
| fn_h | `dyck2`                       | binary (special lengths: `[100,80,60,40,20]`) |
| fn_i | `prime_decimal`               | **decimal** |
| fn_j | `automata_parity`             | binary |
| fn_k | `prime_decimal_tf_check`      | **decimal** |
| fn_l | `sha256_parity`               | binary |

Targets flagged as decimal are listed in `DECIMAL_FNS = {"prime_decimal", "prime_decimal_tf_check"}` and receive a decimal problem statement.

---

## Data generators (high-level)

All generators return a list of dicts:
```python
{'Input': np.array([... as strings ...]), 'Output': '0' or '1'}
```

- **`BinaryDataGenerator`**  
  Balanced 50/50 label split **by construction** against any registered function in `TARGET_FUNCTIONS`.  
  Efficient uniqueness + balancing loop; CPU set-ops for dedup; shuffles final dataset.

- **`PalindromeDataGenerator`**  
  Half palindromes, half non-palindromes (generated by flipping one bit in first half).

- **`PatternBasedDataGenerator`**  
  Balanced presence/absence of a configurable pattern (defaults to `10101010`).  
  Generates with-pattern by insertion; generates without-pattern by “repairing” collisions.

- **`Dyck2DataGenerator`**  
  Balanced valid/invalid Dyck-2 sequences; invalids are near-miss corruptions of valids.  
  Sequence length must be a multiple of 4 (2 bits/paren; pairs become `()[]`).

- **Prime (decimal) families**  
  - `PrimeDataGenerator` — balanced primes vs non-primes (random sampling + `sympy`).  
  - `PrimeDecimalTailRestrictedDataGenerator` — primes vs **non-primes ending with {1,3,7,9}**.  

See **`src/data_handler.py`** for constructor signatures and invariants (e.g., many require `num_samples` to be **even**).

---

## Ground-truth functions registry

`src/target_functions.py` defines many labelers (e.g., parities, automata, pattern matchers, SHA-256 parity) and registers them in:

---

## Evaluation & early stopping

- **Validation**: always computed.  
- **Test**: computed **only if** `val_acc == 1.0` or `val_acc > max_val_acc` for a given attempt; early-stops further attempts for that `(fn, L)` (if `val_acc==1.0`).  
- If no perfect validation, you still get one CSV row per attempt with test accuracy calculated for point with bumps in validation accuracy. You can post select the last test accuracy calculated.

---

## Outputs

### JSONL (one row per attempt)

Each row is a superset; fields include the full prompt and the raw text returned:

```json
{
  "fn": "fn_a",
  "length": 20,
  "attempt": 1,
  "prompt": "...",
  "text": "{\"code\": \"def f(x): ...\"}",
  "usage": { "prompt_tokens": , "completion_tokens": , "reasoning_tokens": , "cached_tokens": },
  "duration_ms": 1234,
  "cached_tokens": 0,
  "tool_uses": 0,
  "tool_results_chars": 0,
  "val_acc": 1.0,
  "test_acc": 1.0,
  "stopped_early": true,
  "compile_error": null
}
```

### CSV columns

- **Identity**: `fn`, `length`, `attempt`
- **Text**: `prompt`, `text` (raw output)
- **Timing/usage**: `duration_ms`, `cached_tokens`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `tool_uses`, `tool_results_chars`
- **Metrics**: `val_acc`, `test_acc`, `stopped_early`, `compile_error`

---

## Design notes & invariants

- **Balancing**: most generators create exactly half positives/negatives to make accuracy comparable and stable.
- **Lengths**: Dyck-2 requires `sequence_length % 4 == 0`; runner special-cases `fn_h`.
- **Decimal vs binary**: affects prompt only; datasets are always rendered as strings: `'0'/'1'` for binary bits, `'0'..'9'` for decimal digits.

---
