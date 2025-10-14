import os
import shutil
import re
import subprocess
from tqdm import tqdm
import time
import main

import os, sys, shutil, subprocess
from tqdm import tqdm
import time

def run_job(results_path):
    env = os.environ.copy()
    # Helps fragmentation and large allocations across steps
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    # Spawn a fresh interpreter to run training
    cmd = [sys.executable, "-u", "main.py", "--results-path", results_path]
    subprocess.run(cmd, check=True, env=env)


def find_max_index(directory):
    items = os.listdir(directory)
    numeric_dirs = [
        item
        for item in items
        if item.isdigit() and os.path.isdir(os.path.join(directory, item))
    ]
    numeric_indices = [int(num) for num in numeric_dirs]
    if numeric_indices:
        return max(numeric_indices)
    else:
        return None

if __name__ == "__main__":
    experiment_idx = find_max_index("./results_new") + 1
    os.mkdir("./results_new/" + str(experiment_idx))

    print (experiment_idx)

    # Specify a base file for all hyperparams
    base_file = input("Please specify filepath of base settings:\n")
    if base_file == "conf/settings.py":
        raise Exception(
            "Choose different filepath for base settings. conf/settings.py is used by sweep to queue experiments."
        )
    if not os.path.isfile(base_file):
        raise Exception("Invalid filepath.")

    grid = {}
    while True:
        hyperparam = input("\nSpecify hyperparam to vary (e.g. lr) or 'END':\n")
        if hyperparam == "END":
            break
        grid[hyperparam] = []

        print("\n\tSpecify value of hyperparam and how many as AxB or 'END':")
        while True:
            value = input("\t")
            if value == "END":
                break
            val_x_num = value.split("x")
            val = val_x_num[0]
            num = int(val_x_num[1])
            for i in range(num):
                grid[hyperparam].append(val)

    # Output and queue the experiments.
    hpms = list(grid)  # list of hyperparams
    count = [0] * len(grid)

    # confirm intent to queue experiments
    num_jobs = 1
    for hp in hpms:
        num_jobs *= len(grid[hp])
    confirm = input(
        "\nYou are about to queue "
        + str(num_jobs)
        + " jobs. Would you like to proceed? (y/n)\n"
    )
    if confirm != "y":
        raise Exception("User failed to confirm.")

    # iterate through all hyperparams
    log = []
    iterate_loop = True
    pbar = tqdm(total=num_jobs)
    model_count = 0
    while iterate_loop:
        shutil.copyfile(base_file, "conf/settings.py")

        for i, hp in enumerate(hpms):
            with open("conf/settings.py", "r") as f:
                lines = f.readlines()

            # Find and replace the parameter value
            for j in range(len(lines)):
                if lines[j].startswith(hp + " = "):
                    lines[j] = hp + " = " + grid[hp][count[i]] + "\n"

            with open("conf/settings.py", "w") as f:
                for line in lines:
                    f.write(line)
        
        current_result = "./results_new/" + str( experiment_idx ) + "/" + str( model_count )
        file_name = "config.txt"
        os.mkdir(current_result)
        shutil.copyfile("./conf/settings.py", os.path.join(current_result, file_name) )
        # main.main( results_path = current_result )
        run_job(current_result)
        model_count += 1
        
        for i, hp in enumerate(hpms):
            log.append("\t" + hp + "=" + grid[hp][count[i]])
        log.append("")

        # counting algorithm
        digit = len(grid) - 1  # 0-indexed
        while True:
            if len(grid)==0:
                iterate_loop = False
                break
            if count[digit] < len(grid[hpms[digit]]) - 1:
                count[digit] += 1
                break
            else:
                count[digit] = 0
                digit -= 1
            if digit == -1:
                iterate_loop = False
                break

        time.sleep(0.5)
        pbar.update(1)

    # write the log
    with open("out_sweep.txt", "w") as f:
        for line in log:
            f.write(line + "\n")
            print(line)

    shutil.copyfile(base_file, "conf/settings.py")