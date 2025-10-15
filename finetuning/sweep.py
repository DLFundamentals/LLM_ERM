# /finetuning/sweep.py

import os
import subprocess
import sys
import json
import argparse
import itertools
from tqdm import tqdm
import yaml

def run_job(cmd, params, results_path):
    """Runs a training job as a subprocess and saves its configuration."""
    config_dict = params

    config_path = os.path.join(results_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    print(f"\n--- Running Job: {' '.join(cmd)} ---")
    log_path = os.path.join(results_path, "stdout.log")
    with open(log_path, 'w') as log_file:
        subprocess.run(cmd, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def main(config_path: str):
    # Load the sweep configuration from the YAML file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    # --- Setup Experiment Directory ---
    experiment_dir = os.path.join(config['results_root'], config['experiment_name'])
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        print(f"Created new experiment directory: {experiment_dir}")
    else:
        print(f"Using existing experiment directory: {experiment_dir}")
        
    # --- Build the list of all hyperparameter combinations ---
    grid_keys = list(config['grid'].keys())
    grid_values = list(config['grid'].values())
    
    job_configs = []
    for bundle in itertools.product(*grid_values):
        current_params = dict(zip(grid_keys, bundle))
        # Merge base arguments with the current grid combination
        # The grid params will override base_args if there are any conflicts
        final_params = {**config['base_args'], **current_params}
        job_configs.append(final_params)
        
    num_jobs = len(job_configs)
    
    # --- User Confirmation ---
    print(f"\n--- Sweep Configuration Summary ---")
    print(f"Experiment Name: {config['experiment_name']}")
    print(f"Results Root: {config['results_root']}")
    print(f"Number of jobs to run: {num_jobs}")
    print(f"------------------------------------")
    
    confirm = input(f"Would you like to proceed? (y/n)\n")
    if confirm.lower() != "y":
        print("Aborted by user.")
        sys.exit(0)

    # --- Execute all jobs ---
    pbar = tqdm(total=num_jobs, desc="Running experiments")
    for i, params in enumerate(job_configs):
        # Create a unique results directory for this specific run
        current_result_path = os.path.join(experiment_dir, str(i))
        if not os.path.exists(current_result_path):
            os.makedirs(current_result_path)

        # # Build the command
        # cmd = [sys.executable, "-u", "main.py", "--results-path", current_result_path]
        # for hp_name, hp_val in params.items():
        #     cmd.append(f"--{hp_name}")
        #     cmd.append(str(hp_val))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_script_path = os.path.join(script_dir, "main.py")

        # Build the command using the correct path
        cmd = [sys.executable, "-u", main_script_path, "--results-path", current_result_path]
        for hp_name, hp_val in params.items():
            cmd.append(f"--{hp_name}")
            cmd.append(str(hp_val))
        
        # Run the job
        try:
            # Pass the `params` dictionary directly to run_job
            run_job(cmd, params, current_result_path)
        except subprocess.CalledProcessError as e:
            print(f"!!! JOB FAILED: {current_result_path} !!!")
            print(f"Error: {e}. Check logs in {os.path.join(current_result_path, 'stdout.log')}")
        
        pbar.update(1)

    pbar.close()
    print(f"\nSweep complete. {num_jobs} jobs were run.")
    print(f"Results are in: {experiment_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep from a YAML config file.")
    parser.add_argument(
        "config_path", 
        type=str, 
        help="Path to the sweep configuration YAML file."
    )
    args = parser.parse_args()
    main(args.config_path)