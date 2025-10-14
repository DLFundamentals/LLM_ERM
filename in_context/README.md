# LLM In-Context Learning Benchmark with vLLM

This code provides a framework for evaluating the in-context learning (ICL) capabilities of Large Language Models (LLMs) on a variety of algorithmic tasks.

The system generates synthetic datasets for tasks like parity checking, primality testing, and formal language recognition. It then constructs few-shot prompts, runs high-throughput inference using a locally-hosted model via [vLLM](https://github.com/vllm-project/vllm), and automatically evaluates the model's accuracy on each task.

## Core Workflow

The experiment process is fully automated:

1.  **Task Definition**: An experiment is defined by a grid of tasks, where each task is a combination of a target function (e.g., `parity_all`) and a sequence length (e.g., `L=50`).
2.  **Prompt Generation**: For each task, the script generates a set of deterministic prompts. Each prompt contains hundreds of few-shot examples and a single test query.
3.  **Batched Inference**: The prompts are sent to a vLLM-powered inference engine for efficient, batched generation.

## Project Structure

-   `vllm_runner.py`: The main entry point and experiment orchestrator. It handles configuration, calls the data generators, runs vLLM inference, and saves results.
-   `../data_handler.py`: Contains a suite of classes for generating balanced (50/50) datasets for all supported tasks. It includes generators for binary sequences, decimal primes, palindromes, and more.
-   `../target_functions.py`: Defines the ground-truth Python functions (e.g., parity checks, pattern matching) used by `data_handler.py` to label the binary data.

## Setup and Installation

1.  **Prerequisites**:
    -   Python 3.8+
    -   An NVIDIA GPU with CUDA installed.
    -   Access to Hugging Face models.

2.  **Installation**:
    Clone the repository and install the required dependencies. It is highly recommended to use a virtual environment.

    ```bash
    # Install vLLM (ensure it matches your CUDA version)
    pip install vllm

    # Install other dependencies
    pip install torch numpy sympy tqdm
    ```

## Usage

The script is executed from the command line. 

### Replicate paper
Here we used three differnt models, Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B-Instruct, and Deepseek-Coder-33B-Instruct. Rest all config are set default in the code.

```bash
python vllm_incontext.py --model <huggingface-model-id>
```

### Play with code

You can specify which models, functions, and sequence lengths to test, along with data and sampling parameters.

```bash
python vllm_runner.py \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --functions fn_a \
    --lengths 20 \
    --train-size 200 \
    --test-size 100 \
    --tensor-parallel-size 1 \
    --out-csv "qwen3_summary.csv" \
    --out-jsonl "qwen3_details.jsonl"
```

### Key Arguments:

-   `--model`: The Hugging Face model ID to use.
-   `--functions`: A list of function IDs to test (see table below).
-   `--lengths`: A list of sequence lengths for the input data.
-   `--train-size`: The number of in-context examples in each prompt.
-   `--test-size`: The number of test prompts to generate per task.
-   `--tensor-parallel-size`: The number of GPUs to use for tensor parallelism.
-   `--out-csv` / `--out-jsonl`: Paths for the output files.

## Output Artifacts

The script generates two output files:

1.  **`*.jsonl`**: A detailed log where each line is a JSON object containing the full prompt, true label, model's raw output, parsed prediction, and correctness for a single test sample.
2.  **`*.csv`**: A summary file containing the final accuracy for each task (function and length combination).