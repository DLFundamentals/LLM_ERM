# =====================================================================================
# runner.py — COMMENTED VERSION
# =====================================================================================
# High‑level purpose
# ------------------
# This script orchestrates an end‑to‑end experiment loop where an OpenAI GPT-5
# model is prompted with synthetic classification data and asked to output a
# Python function that reproduces the target mapping. The code:
#   • Builds persistent train/val/test splits per (target_function, sequence_length).
#   • Calls the OpenAI Responses API asynchronously with concurrency control.
#   • Extracts a candidate function from the model’s JSON output, compiles it,
#     and evaluates its accuracy on validation (and then test, on perfect val).
#   • Performs multiple attempts per grid point and supports early‑stopping when
#     validation accuracy hits 1.0. Else take the test accuracy correspondint to 
#     the maxiumum validation accuracy.
#   • Logs all steps as structured JSON to stdout and to runner.log.
#   • Emits a JSONL and CSV with detailed metrics per attempt.
#
# Some more techinal details:
#   1) Dataset persistence & determinism: For any (fn, L) pair and a global seed,
#      derived_seed = (hash((fn, L)) & 0x7fffffff) ^ global_seed. Splits are saved
#      under datasets/<target>/L<length>/seed<derived_seed>/, so subsequent runs
#      reuse identical data, making results reproducible.
#   2) Prompts demand a single JSON object with the generated function under
#      the "code" key, simplifying parsing.
#   3) Compilation sandbox: we `exec` inside a controlled namespace and pick the
#      function named `f` if present (else the first def). We coerce predictions
#      to 0/1 with bool().
#   4) Early stop: if validation accuracy is exactly 1.0, we compute test and stop.
# =====================================================================================

from __future__ import annotations
import os, sys, json, csv, time, argparse, asyncio, re, ast, textwrap, random, tempfile, shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Mapping, Callable
import logging

from openai import AsyncOpenAI

# --- data generators (unchanged; we orchestrate them here) ---
# These classes must provide a .generate_data() method returning an iterable of
# samples where each sample has keys 'Input' (array of characters '0'/'1') and
# 'Output' ('0' or '1').
from data_handler import (
    BinaryDataGenerator, Dyck2DataGenerator, PatternBasedDataGenerator, 
    PrimeDataGenerator, PalindromeDataGenerator, PrimeDecimalTailRestrictedDataGenerator,
)

external_get_accuracy = None


# =========================
# Usage normalization
# =========================
# Convert various usage payload shapes from the Responses API to a consistent dict.
# This helps when logging and writing CSV/JSONL later.

def normalize_usage(usage_obj) -> Dict[str, Any]:
    if not usage_obj:
        return {}
    if isinstance(usage_obj, Mapping):
        u = dict(usage_obj)
    else:
        u = {}
        for k in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "input_tokens", "output_tokens", "reasoning_tokens"
        ):
            v = getattr(usage_obj, k, None)
            if v is not None:
                u[k] = v

        def _to_dict(obj):
            # Normalize nested token detail objects to plain dicts.
            if obj is None:
                return None
            if isinstance(obj, Mapping):
                return dict(obj)
            d = {}
            for kk in ("cached_tokens", "audio_tokens", "reasoning_tokens"):
                vv = getattr(obj, kk, None)
                if vv is not None:
                    d[kk] = vv
            return d or None

        ptd = getattr(usage_obj, "prompt_tokens_details", None)
        itd = getattr(usage_obj, "input_token_details", None)
        otd = getattr(usage_obj, "output_tokens_details", None)
        ctd = getattr(usage_obj, "completion_tokens_details", None)
        if (ptd := _to_dict(ptd)) is not None:
            u["prompt_tokens_details"] = ptd
        if (itd := _to_dict(itd)) is not None:
            u["input_token_details"] = itd
        if (otd := _to_dict(otd)) is not None:
            u["output_tokens_details"] = otd
        if (ctd := _to_dict(ctd)) is not None:
            u["completion_tokens_details"] = ctd

    # Harmonize field names across different payload versions
    if "prompt_tokens" not in u and "input_tokens" in u:
        u["prompt_tokens"] = u["input_tokens"]
    if "completion_tokens" not in u and "output_tokens" in u:
        u["completion_tokens"] = u["output_tokens"]

    # Bubble up cached_tokens, if present in any detail object
    details = u.get("prompt_tokens_details") or u.get("input_token_details") or {}
    if "cached_tokens" in details:
        u["cached_tokens"] = details["cached_tokens"]

    # Best‑effort reasoning token extraction
    if "reasoning_tokens" not in u or u.get("reasoning_tokens") is None:
        rt = None
        for dkey in ("output_tokens_details", "completion_tokens_details", "input_token_details"):
            d = u.get(dkey) or {}
            if isinstance(d, Mapping):
                rt = d.get("reasoning_tokens")
            if rt is not None:
                break
        if rt is not None:
            u["reasoning_tokens"] = rt

    return u


# =========================
# Logging
# =========================
# We log each event as a single JSON record, both to stdout and runner.log.

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "ts": int(time.time() * 1000),
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        # Include any additional attributes that might be attached via 'extra='
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in base and k not in ("msg", "args", "levelname", "name"):
                base[k] = v
        return json.dumps(base, ensure_ascii=False)

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("runner")
    logger.setLevel(level.upper())

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonFormatter())

    file_handler = logging.FileHandler("runner.log", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())

    # Replace any existing handlers to avoid duplicate logs
    logger.handlers[:] = [stream_handler, file_handler]
    logger.propagate = False
    return logger


# =========================
# Config (now includes dataset_dir)
# =========================
# Central configuration with sensible defaults. Many fields can be overridden via
# CLI flags or environment variables. The OpenAI parameters mirror Responses API
# fields (model, max_output_tokens, reasoning.effort, text.verbosity, tools).

@dataclass
class Config:
    # OpenAI
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = os.getenv("OPENAI_MODEL", "gpt-5")
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "20000"))
    reasoning_effort: str = os.getenv("REASONING_EFFORT", "high")
    verbosity: Optional[str] = os.getenv("TEXT_VERBOSITY", "low")
    tool_choice: str = os.getenv("TOOL_CHOICE", "auto")
    enable_code_interpreter: bool = os.getenv("ENABLE_CODE_INTERPRETER", "0") == "1"

    # Execution
    dry_run: bool = os.getenv("DRY_RUN", "0") == "1"
    concurrency: int = int(os.getenv("CONCURRENCY", "5"))
    per_call_timeout_s: float = float(os.getenv("PER_CALL_TIMEOUT_S", "1200"))

    # Experiment grid
    # The logical functions to target (see FUNCTION_NAME_MAPPING below) and the
    # sequence lengths to evaluate.
    functions: List[str] = field(default_factory=lambda: ["fn_a",
                                                          "fn_b",
                                                          "fn_c",
                                                          "fn_d",
                                                          "fn_e",
                                                          "fn_f",
                                                          "fn_g",
                                                          "fn_h",
                                                          "fn_i",
                                                          "fn_j",
                                                          "fn_k",
                                                          "fn_l"])
    lengths: List[int] = field(default_factory=lambda: [100, 50, 30, 25, 20])
    attempts: int = int(os.getenv("ATTEMPTS", "5"))

    # Dataset settings: fixed sizes for reproducible comparisons across attempts
    train_size: int = int(os.getenv("TRAIN_SIZE", "100"))
    val_size: int   = int(os.getenv("VAL_SIZE",   "100"))
    test_size: int  = int(os.getenv("TEST_SIZE",  "10000"))
    seed: int       = int(os.getenv("GLOBAL_SEED","42"))
    dataset_dir: str = os.getenv("DATASET_DIR", "datasets")

    # Artifacts
    out_jsonl: str = os.getenv("OUT_JSONL", "results_attempts.jsonl")
    out_csv: str   = os.getenv("OUT_CSV",   "results_attempts.csv")


# =========================
# Prompt
# =========================
# Generate the exact user content fed to the model: a problem statement describing
# the data shape and a small batch of examples, followed by strict output format
# instructions (single JSON object with a "code" field).

def build_user_prompt(data_examples: List[str], seq_len: int, decimal: bool = False) -> str:
    if decimal:
        problem_statement = (
            f"**Problem Statement:**\n"
            f"Given a sequence of input vectors (decimal, length {seq_len}) and their corresponding scalar binary outputs ('0' or '1'), "
            f"find a concise Python function `f(x)` that accurately approximates the underlying relationship. "
            f"The function should not be a trainable model, but a direct logical or mathematical representation of the target function."
        )
    else:
        problem_statement = (
            f"**Problem Statement:**\n"
            f"Given a sequence of input vectors (binary, length {seq_len}) and their corresponding scalar binary outputs ('0' or '1'), "
            f"find a concise Python function `f(x)` that accurately approximates the underlying relationship. "
            f"The function should not be a trainable model, but a direct logical or mathematical representation of the target function."
        )
    prompt = f"{problem_statement}\n"
    prompt += "**Data Examples:**\n```\n" + "\n".join(data_examples) + "\n```\n\n"
    prompt += 'You must output ONLY a single JSON object: {"code": "<python function>"}.'
    return prompt


# =========================
# Function mapping & decimal set
# =========================
# Map short experiment IDs (fn_a ... fn_l) to target generator names in data_handler.
# DECIMAL_FNS marks those targets that operate on decimal strings, affecting the
# problem statement.

FUNCTION_NAME_MAPPING = {
    "fn_a": "parity_all",
    "fn_b": "parity_first_half",
    "fn_c": "patternmatch1",
    "fn_d": "patternmatch2",
    "fn_e": "parity_rand_3",
    "fn_f": "parity_rand_10",
    "fn_g": "palindrome",
    "fn_h": "dyck2",
    "fn_i": "prime_decimal",
    "fn_j": "automata_parity",
    "fn_k": "prime_decimal_tf_check",
    "fn_l": "sha256_parity",
}

DECIMAL_FNS = {"prime_decimal", "prime_decimal_tf_check"}


# =========================
# Atomic file helpers
# =========================
# Use write‑to‑temp + os.replace for crash‑safe, atomic writes so partially written
# files are never observed by readers.

def _safe_write_text_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(path)) as tmp:
        for ln in lines:
            tmp.write(f"{ln}\n")
        tmp_path = tmp.name
    os.replace(tmp_path, path)

def _safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(path)) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)

def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


# =========================
# DatasetStore: persists splits under datasets/<target>/L<length>/seed<derived_seed>/
# =========================
# This class wraps deterministic split generation, reuse, and metadata writing.
# It enforces exact sizes for train/val/test; if mismatch is detected, the split
# is rebuilt deterministically from the derived seed.

class DatasetStore:
    """
    Persists and reuses dataset splits. For each (fn, L) we derive a deterministic seed:
      derived_seed = (hash((fn, L)) & 0x7fffffff) ^ cfg.seed

    Directory layout:
      <dataset_dir>/<target_name>/L<length>/seed<derived_seed>/
        - train.txt : 1 sample per line, "sequence -> label"
        - val.txt
        - test.txt
        - meta.json  : sizes, fn, L, seed, decimal
    """
    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log

    @staticmethod
    def _set_seed(seed: int):
        # Seed Python, NumPy, and PyTorch if available, to make generation reproducible.
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _paths(self, target_name: str, L: int, derived_seed: int) -> Dict[str, str]:
        base = os.path.join(self.cfg.dataset_dir, target_name, f"L{L}", f"seed{derived_seed}")
        return {
            "dir": base,
            "train": os.path.join(base, "train.txt"),
            "val":   os.path.join(base, "val.txt"),
            "test":  os.path.join(base, "test.txt"),
            "meta":  os.path.join(base, "meta.json"),
        }

    def _generate_lines(self, target_name: str, L: int, size: int) -> List[str]:
        # Select the correct generator (exactly as your original logic)
        if target_name == 'dyck2':
            if L == 20:
                # IMPORTANT : Only limited samples are possible to make in low dimentions.
                gen = Dyck2DataGenerator(L, size, allow_duplicates=True)
            else:
                gen = Dyck2DataGenerator(L, size)
            dataset = gen.generate_data()
        elif target_name in ['patternmatch1', 'patternmatch2']:
            if target_name == 'patternmatch2':
                gen = PatternBasedDataGenerator(L, size, pattern_string='00111111')
            else:
                gen = PatternBasedDataGenerator(L, size)  # defaults to '10101010'
            dataset = gen.generate_data()
        elif target_name == "palindrome":
            if L == 20:
                gen = PalindromeDataGenerator(L, size, allow_duplicates=True)
            else:
                gen = PalindromeDataGenerator(L, size)
            dataset = gen.generate_data()
        elif target_name == "prime_decimal":
            gen = PrimeDataGenerator(L, size)
            dataset = gen.generate_data()
        elif target_name == "prime_decimal_tf_check":
            gen = PrimeDecimalTailRestrictedDataGenerator(L, size, False)
            dataset = gen.generate_data()
        else:
            gen = BinaryDataGenerator(target_name, L, size)
            dataset = gen.generate_data()

        # Render each sample as "<sequence> -> <label>" line for simple evaluation later.
        return [f"{''.join(sample['Input'].tolist())} -> {sample['Output']}" for sample in dataset]

    def _ensure_splits(self, fn: str, L: int) -> Tuple[List[str], List[str], List[str], bool]:
        target_name = FUNCTION_NAME_MAPPING[fn]
        is_decimal = target_name in DECIMAL_FNS

        import hashlib
        # The SGD script splits into train/test, so we combine val+train sizes for its 'train' part.
        sgd_equivalent_train_size = self.cfg.train_size + self.cfg.val_size
        sgd_equivalent_test_size = self.cfg.test_size
        hash_str = (
            f"func={target_name}-len={L}-"
            f"train={sgd_equivalent_train_size}-test={sgd_equivalent_test_size}"
        )

        h = hashlib.sha256(hash_str.encode()).digest()
        derived_seed = (int.from_bytes(h, 'big') & 0x7fffffff) ^ self.cfg.seed
        paths = self._paths(target_name, L, derived_seed)
        os.makedirs(paths["dir"], exist_ok=True)

        def exists_with_size(path: str, expect: int) -> bool:
            # Guard against truncated files or wrong sizes by counting lines.
            if not os.path.exists(path):
                return False
            try:
                cnt = sum(1 for _ in open(path, "r", encoding="utf-8"))
                return cnt == expect
            except Exception:
                return False

        # If all exist with exact sizes, reuse from cache
        if (exists_with_size(paths["train"], self.cfg.train_size) and
            exists_with_size(paths["val"],   self.cfg.val_size)   and
            exists_with_size(paths["test"],  self.cfg.test_size)):
            train_lines = _read_lines(paths["train"])
            val_lines   = _read_lines(paths["val"])
            test_lines  = _read_lines(paths["test"])
            self.log.info("dataset_reused", extra={"fn": fn, "length": L, "seed": derived_seed, "dir": paths["dir"]})
            return train_lines, val_lines, test_lines, is_decimal

        # Otherwise (re)generate deterministically and write atomically
        self._set_seed(derived_seed)
        self.log.info("dataset_generating", extra={"fn": fn, "length": L, "seed": derived_seed, "dir": paths["dir"]})

        # 1. Generate ONE large, ORDERED pool of data.
        total_samples_for_pool = self.cfg.train_size + self.cfg.val_size + self.cfg.test_size
        all_lines = self._generate_lines(target_name, L, total_samples_for_pool)

        # 2. Separate the positive and negative samples based on their initial order.
        num_pos_total = total_samples_for_pool // 2
        
        pos_lines = all_lines[:num_pos_total]
        neg_lines = all_lines[num_pos_total:]
        
        # 3. Perform a deterministic shuffle ON EACH CLASS SEPARATELY.
        # This mimics the SGD script's `torch.randperm` on the indices of each class.
        random.shuffle(pos_lines)
        random.shuffle(neg_lines)

        # 4. Perform the stratified split by slicing the shuffled class lists.
        # This guarantees a 50/50 balance in train and validation sets.
        train_pos_count = self.cfg.train_size // 2
        train_neg_count = self.cfg.train_size - train_pos_count
        
        val_pos_count = self.cfg.val_size // 2
        val_neg_count = self.cfg.val_size - val_pos_count

        # Combine the slices to create the train, val, and test sets
        train_lines = pos_lines[:train_pos_count] + neg_lines[:train_neg_count]
        
        val_lines = (
            pos_lines[train_pos_count : train_pos_count + val_pos_count] + 
            neg_lines[train_neg_count : train_neg_count + val_neg_count]
        )
                    
        test_lines = (
            pos_lines[train_pos_count + val_pos_count:] + 
            neg_lines[train_neg_count + val_neg_count:]
        )

        # 5. Final shuffle of the combined train and validation sets before saving.
        # This mixes the positive and negative samples within each split.
        random.shuffle(train_lines)
        random.shuffle(val_lines)
        # The test set is not shuffled

        # 6. Sanity checks for sizes
        assert len(train_lines) == self.cfg.train_size
        assert len(val_lines) == self.cfg.val_size
        assert len(test_lines) == self.cfg.test_size

        # 7. Write the final splits atomically to disk
        _safe_write_text_lines(paths["train"], train_lines)
        _safe_write_text_lines(paths["val"],   val_lines)
        _safe_write_text_lines(paths["test"],  test_lines)
        _safe_write_json(paths["meta"], {
            "fn": fn,
            "target_name": target_name,
            "length": L,
            "decimal": is_decimal,
            "derived_seed": derived_seed,
            "hash_str_used": hash_str,
            "sizes": {"train": self.cfg.train_size, "val": self.cfg.val_size, "test": self.cfg.test_size},
            "created_ts": int(time.time())
        })

        self.log.info("dataset_written", extra={"fn": fn, "length": L, "seed": derived_seed, "dir": paths["dir"]})
        return train_lines, val_lines, test_lines, is_decimal

    def get(self, fn: str, L: int) -> Tuple[List[str], List[str], List[str], bool]:
        """
        Returns (train, val, test, is_decimal), pulling from disk or generating once.
        """
        return self._ensure_splits(fn, L)


# =========================
# Code extraction & compilation
# =========================
# The model is instructed to return a *single* JSON object {"code": "..."}. We:
#   • Parse JSON strictly first; if that fails, fallback to a broad { ... } regex.
#   • Sanitize fenced code blocks and dedent before AST parsing.
#   • Choose function named 'f' if present; else the first defined function.
#   • Execute in a restricted global namespace and retrieve the callable.

def extract_code_from_output(output_text: str) -> Optional[str]:
    if not output_text:
        return None
    try:
        obj = json.loads(output_text)
        if isinstance(obj, dict) and "code" in obj and isinstance(obj["code"], str):
            return obj["code"]
    except Exception:
        pass
    m = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "code" in obj and isinstance(obj["code"], str):
                return obj["code"]
        except Exception:
            return None
    return None

def compile_callable_from_code(code_str: str) -> Callable[[str], int]:
    code_str = textwrap.dedent(code_str.strip())
    if code_str.startswith("```"):
        # Remove markdown fences like ```python ... ``` if present
        code_str = re.sub(r"^```(?:python)?\s*|\s*```$", "", code_str, flags=re.IGNORECASE | re.DOTALL)
    tree = ast.parse(code_str)
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_names:
        raise ValueError("No function definition found in generated code.")
    prefer_name = "f" if "f" in fn_names else fn_names[0]
    local_ns: Dict[str, Any] = {}
    safe_globals = {"__builtins__": __builtins__}
    exec(compile(tree, filename="<generated>", mode="exec"), safe_globals, local_ns)
    fn = local_ns.get(prefer_name)
    if not callable(fn):
        raise ValueError(f"Function '{prefer_name}' not found after exec.")
    return fn


# =========================
# Accuracy evaluation
# =========================
# Evaluate a callable against a list of "<sequence> -> <label>" lines. We default
# to a local implementation but can delegate to external_get_accuracy if present.

def _local_get_accuracy(fn_callable: Callable[[str], int], data_lines: List[str]) -> float:
    if not data_lines:
        return 0.0
    correct = 0
    for line in data_lines:
        try:
            x, y = line.split("->")
            x = x.strip()
            y = y.strip()
            y_int = int(y)
            pred = fn_callable(x)
            pred_int = int(bool(pred))
            correct += int(pred_int == y_int)
        except Exception:
            # Silently ignore malformed lines or runtime errors in user code.
            pass
    return correct / len(data_lines)

def evaluate_accuracy(fn_callable: Callable[[str], int], data_lines: List[str]) -> float:
    if external_get_accuracy is not None:
        try:
            return float(external_get_accuracy(fn_callable, data_lines))
        except Exception:
            pass
    return _local_get_accuracy(fn_callable, data_lines)


# =========================
# Runner
# =========================
# The Runner coordinates dataset retrieval, API calls, code extraction/compilation,
# evaluation, early stopping, logging, and artifact writing.

class Runner:
    def __init__(self, cfg: Config, logger: logging.Logger):
        if not cfg.api_key:
            raise SystemExit("OPENAI_API_KEY is required.")
        self.cfg = cfg
        self.log = logger
        self.client = AsyncOpenAI(api_key=cfg.api_key)
        self.sem = asyncio.Semaphore(cfg.concurrency)

        # Optional tool injection (Code Interpreter) for the Responses API.
        self.tools: List[Dict[str, Any]] = []
        if cfg.enable_code_interpreter:
            self.tools.append({"type": "code_interpreter", "container": {"type": "auto"}})

        # Persisted dataset store
        self.ds = DatasetStore(cfg, logger)

    async def _call_once(self, fn: str, L: int, attempt_idx: int, data_examples: List[str], decimal: bool) -> Dict[str, Any]:
        # Prepare prompt and request body for one model attempt
        prompt_text = build_user_prompt(data_examples, L, decimal)
        body_preview_size = len(json.dumps({"input":[{"role":"user","content":[{"type":"input_text","text": prompt_text}]}]}))
        body: Dict[str, Any] = {
            "model": self.cfg.model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
            "reasoning": {"effort": self.cfg.reasoning_effort},
            "max_output_tokens": self.cfg.max_output_tokens,
            "tool_choice": self.cfg.tool_choice,
        }
        if self.tools:
            body["tools"] = self.tools
        if self.cfg.verbosity:
            body["text"] = {"verbosity": self.cfg.verbosity}

        if self.cfg.dry_run:
            # In dry‑run, just log the prompt and return a lightweight record.
            self.log.info("dry_run_input", extra={"fn": fn, "length": L, "attempt": attempt_idx, "prompt_preview": prompt_text})
            return {
                "fn": fn, "length": L, "attempt": attempt_idx,
                "prompt": prompt_text, "request_body": body,
                "text": None, "usage": {}, "cached_tokens": 0, "duration_ms": 0
            }

        async def _try_call(tag: str):
            # Execute a single Responses API call with a timeout and collect rich telemetry
            t0 = time.perf_counter()
            async with self.sem:
                res = await asyncio.wait_for(
                    self.client.responses.create(**body),
                    timeout=self.cfg.per_call_timeout_s,
                )
                # Count how many tool_use/tool_result chunks came back
                tool_uses = 0
                tool_results_chars = 0
                for item in getattr(res, "output", []) or []:
                    t = getattr(item, "type", None)
                    if t == "tool_use":
                        tool_uses += 1
                    elif t == "tool_result":
                        content = getattr(item, "content", None)
                        if isinstance(content, list):
                            for part in content:
                                txt = getattr(part, "text", None) if hasattr(part, "text") else part.get("text") if isinstance(part, dict) else None
                                if isinstance(txt, str):
                                    tool_results_chars += len(txt)
                        elif isinstance(content, str):
                            tool_results_chars += len(content)

            dt_ms = int((time.perf_counter() - t0) * 1000)
            usage = normalize_usage(getattr(res, "usage", {}))
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            out_text = (getattr(res, "output_text", "") or "").strip()
            self.log.info(tag, extra={
                "fn": fn, "length": L, "attempt": attempt_idx,
                "duration_ms": dt_ms, "prompt_chars": len(prompt_text),
                "request_body_bytes": len(json.dumps(body)),
                "input_section_bytes": body_preview_size,
                "tools_enabled": bool(self.tools), "tool_count": len(self.tools),
                "tool_uses": tool_uses, "tool_results_chars": tool_results_chars,
                "prompt_tokens": usage.get("prompt_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
                "cached_tokens": cached, "completion_tokens": usage.get("completion_tokens"),
            })
            return {
                "fn": fn, "length": L, "attempt": attempt_idx,
                "prompt": prompt_text,
                "text": out_text, "usage": usage,
                "cached_tokens": cached, "duration_ms": dt_ms,
                "request_body_bytes": len(json.dumps(body)),
                "prompt_chars": len(prompt_text),
                "tool_uses": tool_uses,
                "tool_results_chars": tool_results_chars,
            }

        try:
            return await _try_call("attempt_ok")
        except Exception as e1:
            # One retry for transient failures (timeouts, throttling, etc.)
            self.log.warning("attempt_retry_once", extra={"fn": fn, "length": L, "attempt": attempt_idx, "error": str(e1)})
            try:
                return await _try_call("attempt_ok_after_retry")
            except Exception as e2:
                self.log.error("attempt_failed", extra={"fn": fn, "length": L, "attempt": attempt_idx, "error": str(e2)})
                return {"fn": fn, "length": L, "attempt": attempt_idx, "error": str(e2)}

    async def run(self) -> List[Dict[str, Any]]:
        # Full experiment loop over all (function, length) grid points.
        all_rows: List[Dict[str, Any]] = []

        for fn in self.cfg.functions:
            if fn not in FUNCTION_NAME_MAPPING:
                self.log.error("unknown_function", extra={"fn": fn})
                continue
            
            current_lengths = self.cfg.lengths
            if fn == "fn_h":
                current_lengths = [100, 80, 60, 40, 20]
            for L in current_lengths:
                # 1) persistent dataset
                train_lines, val_lines, test_lines, is_decimal = self.ds.get(fn, L)

                stopped_early = False

                # 2) attempts (sequential to allow early stop)
                for k in range(1, self.cfg.attempts + 1):
                    res = await self._call_once(fn, L, k, train_lines, is_decimal)
                    out_text = res.get("text") or ""
                    code_str = extract_code_from_output(out_text)

                    max_val_acc = 0
                    val_acc = None
                    test_acc = None
                    compile_error = None

                    if code_str:
                        try:
                            fn_callable = compile_callable_from_code(code_str)
                            val_acc = evaluate_accuracy(fn_callable, val_lines)
                            if val_acc >= max_val_acc:
                                max_val_acc = val_acc
                                test_acc = evaluate_accuracy(fn_callable, test_lines)
                                if val_acc == 1.0:
                                    stopped_early = True
                        except Exception as e:
                            compile_error = str(e)
                            self.log.warning("compile_or_eval_error", extra={"fn": fn, "length": L, "attempt": k, "error": compile_error})
                    else:
                        compile_error = "no_code_found"

                    row = {
                        **res,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        "stopped_early": stopped_early,
                        "compile_error": compile_error,
                    }
                    all_rows.append(row)

                    if stopped_early:
                        self.log.info("early_stop", extra={"fn": fn, "length": L, "attempt": k, "val_acc": val_acc, "test_acc": test_acc})
                        break

        self.log.info("dispatch_finished", extra={"total_results": len(all_rows)})
        return all_rows


# =========================
# Writers
# =========================
# Emit a JSONL (1 row per attempt) and a CSV with a fixed set of columns. The CSV
# aligns with normalize_usage fields, making it easy to analyze in spreadsheets.

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "fn", "length", "attempt", "prompt", "text",
        "duration_ms", "cached_tokens", "prompt_tokens", "completion_tokens",
        "reasoning_tokens", "tool_uses", "tool_results_chars",
        "val_acc", "test_acc", "stopped_early", "compile_error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            usage = r.get("usage") or {}
            w.writerow({
                "fn": r.get("fn"),
                "length": r.get("length"),
                "attempt": r.get("attempt"),
                "prompt": r.get("prompt"),
                "text": r.get("text"),
                "duration_ms": r.get("duration_ms"),
                "cached_tokens": r.get("cached_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "tool_uses": r.get("tool_uses"),
                "tool_results_chars": r.get("tool_results_chars"),
                "val_acc": r.get("val_acc"),
                "test_acc": r.get("test_acc"),
                "stopped_early": r.get("stopped_early"),
                "compile_error": r.get("compile_error"),
            })


# =========================
# CLI
# =========================
# Argparse surface mirrors the Config dataclass. Unknowns default to env vars.
# LOG_LEVEL is passed through the environment so setup_logger can pick it up.

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="OpenAI GPT-5 runner (early-stop + persistent datasets)")

    # Grid / behavior
    p.add_argument("--functions", nargs="*", help="Function IDs (e.g., fn_a fn_b ...)")
    p.add_argument("--lengths", nargs="*", type=int, help="Sequence lengths (e.g., 100 50 30 25 20)")
    p.add_argument("--attempts", type=int, help="Attempts per (fn, length), default=5")
    p.add_argument("--concurrency", type=int, help="Max concurrent API calls (default: 5)")
    p.add_argument("--timeout", type=float, help="Per-call timeout seconds (default: 1200)")

    # OpenAI
    p.add_argument("--model", help="Model name (default: gpt-5)")
    p.add_argument("--max-output-tokens", type=int, help="Max output tokens (default: 20000)")
    p.add_argument("--enable-code-interpreter", action="store_true", help="Enable Code Interpreter tool")
    p.add_argument("--tool-choice", choices=["auto","none"], help="Tool choice (default: auto)")
    p.add_argument("--verbosity", choices=["low","medium","high"], help="text.verbosity (default: low)")
    p.add_argument("--reasoning-effort", choices=["minimal","medium","high"], help="reasoning.effort (default: high)")

    # Datasets
    p.add_argument("--train-size", type=int, help="Train size per (fn, L) (default: 100)")
    p.add_argument("--val-size", type=int, help="Validation size per (fn, L) (default: 100)")
    p.add_argument("--test-size", type=int, help="Test size per (fn, L) (default: 10000)")
    p.add_argument("--seed", type=int, help="Global seed (default: 432)")
    p.add_argument("--dataset-dir", type=str, help="Dataset root directory (default: datasets)")

    # Artifacts
    p.add_argument("--out-jsonl", help="Output JSONL path (default: results_attempts.jsonl)")
    p.add_argument("--out-csv", help="Output CSV path (default: results_attempts.csv)")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logging level (default: INFO)")
    p.add_argument("--dry-run", action="store_true", help="Dry run, shows input prompt generated for each query")

    args = p.parse_args()
    cfg = Config()

    # Apply overrides
    if args.functions: cfg.functions = args.functions
    if args.lengths: cfg.lengths = args.lengths
    if args.attempts: cfg.attempts = args.attempts
    if args.concurrency: cfg.concurrency = args.concurrency
    if args.model: cfg.model = args.model
    if args.max_output_tokens: cfg.max_output_tokens = args.max_output_tokens
    if args.enable_code_interpreter: cfg.enable_code_interpreter = True
    if args.tool_choice: cfg.tool_choice = args.tool_choice
    if args.out_jsonl: cfg.out_jsonl = args.out_jsonl
    if args.out_csv: cfg.out_csv = args.out_csv
    if args.verbosity: cfg.verbosity = args.verbosity
    if args.reasoning_effort: cfg.reasoning_effort = args.reasoning_effort
    if args.timeout: cfg.per_call_timeout_s = args.timeout
    if args.train_size: cfg.train_size = args.train_size
    if args.val_size: cfg.val_size = args.val_size
    if args.test_size: cfg.test_size = args.test_size
    if args.seed is not None: cfg.seed = args.seed
    if args.dataset_dir: cfg.dataset_dir = args.dataset_dir
    if args.dry_run: cfg.dry_run = True

    os.environ["LOG_LEVEL"] = args.log_level
    return cfg


async def _amain(cfg: Config) -> None:
    # Entrypoint for asyncio: build logger, run the experiment, then write artifacts.
    log = setup_logger(os.getenv("LOG_LEVEL", "INFO"))
    runner = Runner(cfg, log)
    try:
        rows = await runner.run()
    finally:
        # Close the AsyncOpenAI client to release network resources.
        await runner.client.close()
    write_jsonl(cfg.out_jsonl, rows)
    write_csv(cfg.out_csv, rows)
    log.info("artifacts_written", extra={"jsonl": cfg.out_jsonl, "csv": cfg.out_csv})


def main() -> None:
    cfg = parse_args()
    asyncio.run(_amain(cfg))


if __name__ == "__main__":
    main()
