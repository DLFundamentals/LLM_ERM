# vllm_runner.py
"""
This script orchestrates an end-to-end experiment loop for in-context learning
using a local model served by vLLM.

The process is as follows:
  1. For each task (a combination of a target function and sequence length),
     it generates a deterministic set of prompts. Each prompt contains a number
     of few-shot examples (the "in-context" part) and one test query.
  2. It uses vLLM to run batched inference on these prompts.
  3. It parses the model's JSON response (e.g., {"label": "1"}) to extract the
     predicted label.
  4. It compares the prediction to the ground truth to calculate accuracy.
  5. It logs results and saves a detailed JSONL file with per-sample outcomes
     and a final CSV summary with accuracy per task.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Ensure vLLM is available
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM is not installed. Please install it with: pip install vllm")
    exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the data generators from the provided project structure
from data_handler import (
    BaseDataGenerator,
    BinaryDataGenerator,
    Dyck2DataGenerator,
    PalindromeDataGenerator,
    PatternBasedDataGenerator,
    PrimeDataGenerator,
    PrimeDecimalTailRestrictedDataGenerator,
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================
@dataclass
class Config:
    """Manages all configuration for the experiment."""
    # Experiment Grid
    functions: List[str] = field(default_factory=lambda: ["fn_a", "fn_b", "fn_c", "fn_d", "fn_e", "fn_f", "fn_g", "fn_h", "fn_i", "fn_j", "fn_k", "fn_l"])
    lengths: List[int] = field(default_factory=lambda: [20, 25, 30, 50, 100])

    # Data Generation
    train_size: int = 200  # Number of in-context examples per prompt
    test_size: int = 100   # Number of test prompts per task (fn, L)
    seed: int = 42

    # vLLM & Model
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # model: str = "deepseek-ai/deepseek-coder-33b-instruct"
    # model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

    tensor_parallel_size: int = 1
    max_model_len: int = 25000
    trust_remote_code: bool = True

    # Sampling Parameters
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 1024

    # Artifacts
    output_jsonl: str = "vllm_results_details.jsonl"
    output_csv: str = "vllm_results_summary.csv"


# =========================
# Constants (from runner.py for consistency)
# =========================
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
# Prompt Generation
# =========================
def build_user_prompt(
    data_examples: List[str],
    test_input: str,
    seq_len: int,
    decimal: bool = False,
) -> str:
    """Creates a structured few-shot prompt for an in-context classification task."""
    problem_type = "decimal" if decimal else "binary"
    problem_statement = (
        f"**Problem Statement:**\n"
        f"You are given input vectors (type: {problem_type}, length: {seq_len}) and their "
        f"corresponding binary outputs ('0' or '1'). Your task is to analyze the provided "
        f"examples to understand the underlying pattern or function. Then, for the final "
        f"test input, predict its correct label."
    )
    prompt = f"{problem_statement}\n\n"
    prompt += "**Data Examples:**\n```\n" + "\n".join(data_examples) + "\n```\n\n"
    prompt += "**Test Input:**\n```\n" + test_input + "\n```\n\n"
    prompt += 'Based on the examples, what is the label for the test input? You must output ONLY a single JSON object in the format: {"label": "<your predicted label>"}.'
    return prompt


class PromptGenerator:
    """Generates deterministic and reproducible prompts for each task."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _get_data_generator(self, target_name: str, L: int, size: int) -> BaseDataGenerator:
        """Selects and instantiates the correct data generator for a given task."""
        # This logic is adapted from your scripts to use the available generators
        if target_name == 'dyck2':
            # Dyck-2 generator has constraints on sample size for smaller lengths
            if L == 20:
                return Dyck2DataGenerator(L, size, allow_duplicates=True)
            else:
                return Dyck2DataGenerator(L, size)
        if target_name in ['patternmatch1', 'patternmatch2']:
            if target_name == 'patternmatch2':
                return PatternBasedDataGenerator(L, size, pattern_string='00111111')
            else:
                return PatternBasedDataGenerator(L, size)  # defaults to '10101010'
        if target_name == "palindrome":
            return PalindromeDataGenerator(L, size)
        if target_name == "prime_decimal":
            return PrimeDataGenerator(L, size)
        if target_name == "prime_decimal_tf_check":
            return PrimeDecimalTailRestrictedDataGenerator(L, size, allow_leading_zeros=False)
        # Default to BinaryDataGenerator for all other function names
        if target_name in FUNCTION_NAME_MAPPING.values():
            return BinaryDataGenerator(target_name, L, size)

        raise ValueError(f"No data generator found for target function '{target_name}'")

    def _generate_lines(self, generator: BaseDataGenerator, target_name: str) -> List[str]:
        """Runs the data generator and formats the output into lines."""
        data = generator.generate_data()
        return [f"{''.join(sample['Input'])} -> {sample['Output']}" for sample in data]

    def generate_prompts_for_task(self, fn: str, L: int) -> List[Dict[str, Any]]:
        """
        Generates a list of fully-formed prompts for a given (fn, L) task.
        Each prompt includes training examples and a single unique test query.
        """
        if fn not in FUNCTION_NAME_MAPPING:
            logger.warning(f"Function key '{fn}' not in FUNCTION_NAME_MAPPING. Skipping.")
            return []

        target_name = FUNCTION_NAME_MAPPING[fn]
        is_decimal = target_name in DECIMAL_FNS

        # Use a derived seed for deterministic data generation, same as in runner.py
        derived_seed = (hash((fn, L)) & 0x7FFFFFFF) ^ self.cfg.seed
        random.seed(derived_seed)

        logger.info(f"Generating data for task (fn={fn}, L={L}) with seed {derived_seed}...")
        try:
            # Generate in-context examples (train) and test queries
            train_gen = self._get_data_generator(target_name, L, self.cfg.train_size)
            test_gen = self._get_data_generator(target_name, L, self.cfg.test_size)
            train_lines = self._generate_lines(train_gen, target_name)
            test_lines = self._generate_lines(test_gen, target_name)
        except Exception as e:
            logger.error(f"Failed to generate data for (fn={fn}, L={L}): {e}", exc_info=True)
            return []

        # Create a prompt for each test line
        prompts = []
        for test_line in test_lines:
            test_input, true_label = [part.strip() for part in test_line.split("->")]
            prompt_text = build_user_prompt(train_lines, test_input, L, is_decimal)
            prompts.append({
                "prompt": prompt_text,
                "true_label": true_label,
                "fn": fn,
                "length": L
            })
        return prompts


# =========================
# VLLM Runner
# =========================
class VLLMRunner:
    """Wraps the vLLM engine and orchestrates the inference process."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prompt_generator = PromptGenerator(cfg)
        logger.info(f"Initializing vLLM engine for model: {cfg.model}")
        try:
            self.llm = LLM(
                model=cfg.model,
                tensor_parallel_size=cfg.tensor_parallel_size,
                trust_remote_code=cfg.trust_remote_code,
                max_model_len=cfg.max_model_len
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM LLM engine: {e}", exc_info=True)
            raise
        self.sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens
        )
        logger.info("vLLM engine initialized successfully.")

    def run_experiment(self) -> List[Dict[str, Any]]:
        """Executes the full experiment across all specified tasks."""
        all_results = []
        for fn in self.cfg.functions:
            if fn == "fn_h":
                current_lengths = [100, 80, 60, 40, 20]
            else:
                current_lengths = self.cfg.lengths
            for L in current_lengths:
                task_prompts = self.prompt_generator.generate_prompts_for_task(fn, L)
                if not task_prompts:
                    continue

                logger.info(f"Running inference for task (fn={fn}, L={L}) with {len(task_prompts)} prompts...")
                start_time = time.perf_counter()

                prompts_to_run = [p['prompt'] for p in task_prompts]
                request_outputs = self.llm.generate(prompts_to_run, self.sampling_params, use_tqdm=True)

                duration = time.perf_counter() - start_time
                throughput = len(prompts_to_run) / duration
                logger.info(f"Task (fn={fn}, L={L}) completed in {duration:.2f}s ({throughput:.2f} prompts/s).")

                # Combine prompts with outputs for evaluation
                for prompt_data, output in zip(task_prompts, request_outputs):
                    model_output_text = output.outputs[0].text.strip()
                    all_results.append({
                        **prompt_data,
                        "model_output": model_output_text
                    })
        return all_results


# =========================
# Evaluation & Artifacts
# =========================
def parse_and_evaluate(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Parses model outputs, evaluates correctness, and computes summary stats.
    This version uses regex to robustly find JSON within extraneous text.
    """
    task_stats = {}
    for res in results:
        task_key = (res['fn'], res['length'])
        if task_key not in task_stats:
            task_stats[task_key] = {'correct': 0, 'total': 0, 'failed_parses': 0}

        pred_label = None
        model_output = res['model_output']

        try:
            # 1. Use regex to find a JSON object within the model's output string.
            #    This handles cases where the model adds introductory/concluding text.
            #    re.DOTALL makes '.' match newlines, in case the JSON is multi-line.
            match = re.search(r'\{.*\}', model_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                if isinstance(data, dict) and 'label' in data:
                    pred_label = str(data['label']).strip()
        except (json.JSONDecodeError, TypeError):
            # This block is reached if regex found something that looked like JSON
            # (e.g., "{...}") but it wasn't valid. We'll let it fall through
            # to the next fallback method.
            pass

        # 2. If JSON parsing failed, try a simpler fallback for raw "0" or "1".
        if pred_label is None:
            clean_output = model_output.strip()
            # Check if the very last non-whitespace character is a 0 or 1
            if clean_output and clean_output[-1] in ["0", "1"]:
                # To be safer, check if the whole string is just "0" or "1"
                # after cleaning quotes. This was the original logic.
                simple_clean = clean_output.replace("'", "").replace('"', '')
                if simple_clean in ["0", "1"]:
                    pred_label = simple_clean

        res['predicted_label'] = pred_label
        res['is_correct'] = (pred_label is not None and pred_label == res['true_label'])

        task_stats[task_key]['total'] += 1
        if res['is_correct']:
            task_stats[task_key]['correct'] += 1
        if pred_label is None:
            task_stats[task_key]['failed_parses'] += 1

    # Calculate accuracy, ignoring tasks with no successful parses if desired
    accuracies = {}
    for (fn, L), stats in task_stats.items():
        total_valid = stats['total']
        if total_valid > 0:
            accuracies[f"{fn}_L{L}"] = stats['correct'] / total_valid
        else:
            accuracies[f"{fn}_L{L}"] = 0.0
        
        if stats['failed_parses'] > 0:
            logger.warning(
                f"Task (fn={fn}, L={L}): Failed to parse "
                f"{stats['failed_parses']}/{stats['total']} outputs."
            )

    return results, accuracies

def save_artifacts(cfg: Config, results: List[Dict[str, Any]], accuracies: Dict[str, float]):
    """Saves detailed JSONL results and a CSV summary."""
    # Save detailed JSONL
    logger.info(f"Saving {len(results)} detailed results to {cfg.output_jsonl}...")
    with open(cfg.output_jsonl, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # Save summary CSV
    logger.info(f"Saving summary accuracies to {cfg.output_csv}...")
    summary_rows = []
    for task_key, acc in accuracies.items():
        fn, L_str = task_key.split('_L')
        summary_rows.append({'function': fn, 'length': int(L_str), 'accuracy': acc})

    with open(cfg.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['function', 'length', 'accuracy'])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Artifacts saved successfully.")


# =========================
# CLI & Main Execution
# =========================
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run in-context learning experiments with vLLM.")
    cfg = Config()

    # Grid
    p.add_argument("--functions", nargs="*", help=f"Function IDs to test (default: {cfg.functions})")
    p.add_argument("--lengths", nargs="*", type=int, help=f"Sequence lengths to test (default: {cfg.lengths})")
    
    # Data
    p.add_argument("--train-size", type=int, help=f"In-context examples per prompt (default: {cfg.train_size})")
    p.add_argument("--test-size", type=int, help=f"Test prompts per task (default: {cfg.test_size})")
    p.add_argument("--seed", type=int, help=f"Global random seed (default: {cfg.seed})")

    # Model & VLLM
    p.add_argument("--model", type=str, help=f"Hugging Face model ID (default: {cfg.model})")
    p.add_argument("--tensor-parallel-size", type=int, help=f"GPU tensor parallelism (default: {cfg.tensor_parallel_size})")
    p.add_argument("--max-model-len", type=int, help=f"Max model sequence length (default: {cfg.max_model_len})")

    # Sampling
    p.add_argument("--temperature", type=float, help=f"Sampling temperature (default: {cfg.temperature})")
    p.add_argument("--max-new-tokens", type=int, help=f"Max generated tokens (default: {cfg.max_new_tokens})")

    # Output
    p.add_argument("--out-jsonl", help="Output JSONL path (default: results_attempts.jsonl)")
    p.add_argument("--out-csv", help="Output CSV path (default: results_attempts.csv)")

    args = p.parse_args()

    # Apply overrides from CLI
    if args.functions: cfg.functions = args.functions
    if args.lengths: cfg.lengths = args.lengths
    if args.train_size: cfg.train_size = args.train_size
    if args.test_size: cfg.test_size = args.test_size
    if args.seed: cfg.seed = args.seed
    if args.model: cfg.model = args.model
    if args.tensor_parallel_size: cfg.tensor_parallel_size = args.tensor_parallel_size
    if args.max_model_len: cfg.max_model_len = args.max_model_len
    if args.temperature is not None: cfg.temperature = args.temperature
    if args.out_jsonl: cfg.output_jsonl = args.out_jsonl
    if args.out_csv: cfg.output_csv = args.out_csv

    return cfg


def main():
    """Main entry point for the script."""
    config = parse_args()
    logger.info(f"Starting experiment with configuration: {config}")

    runner = VLLMRunner(config)
    results = runner.run_experiment()
    
    if not results:
        logger.warning("No results were generated. Exiting.")
        return

    evaluated_results, accuracies = parse_and_evaluate(results)
    
    logger.info("--- Experiment Summary ---")
    for task, acc in accuracies.items():
        logger.info(f"Task: {task:<20} Accuracy: {acc:.2%}")
    logger.info("--------------------------")
    
    save_artifacts(config, evaluated_results, accuracies)


if __name__ == "__main__":
    main()