import os
import operator
import importlib

import random
import numpy as np
import torch

def set_seed(seed: int):
    """Sets the seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def print_performance(epoch, train_acc, test_acc):
    print(f'################## Epoch {epoch} ##################')
    print(f'##### Train: epoch={epoch}: Acc={train_acc:.4f}')
    print(f'##### Test: epoch={epoch}: Acc={test_acc:.4f}')
    print(f'#############################################')

def save_data(dir_name, metrics):
    # Ensure the directory exists
    os.makedirs(dir_name, exist_ok=True)
    
    # Fetch non-callable attributes that do not start with '__'
    attrbts = [attr for attr in dir(metrics) if not callable(getattr(metrics, attr)) and not attr.startswith("__")]

    # Open a single Python file to write all attributes as variables
    with open(os.path.join(dir_name, "metrics.py"), 'w') as file:
        for name in attrbts:
            value = operator.attrgetter(name)(metrics)
            # Write the variable and its value in a way that can be imported as Python code
            file.write(f"{name} = {repr(value)}\n")

def get_dir_name(directory, prespecified=False, resume=False):
    if not prespecified:
        if not os.path.isdir(directory):
            os.mkdir(directory)

        # results directories
        sub_dirs_ids = [int(dir) for dir in os.listdir(directory)
                        if os.path.isdir(directory + '/' + dir)]

        # experiment id
        xid = max(sub_dirs_ids)
        dir_name = directory + '/' + str(xid)
    else:
        dir_name = directory

    # sweeps the INNER directories
    sub_dirs_ids = [int(dir) for dir in os.listdir(dir_name)
                    if os.path.isdir(dir_name + '/' + dir)]

    # current sweep
    if len(sub_dirs_ids) == 0: pid = 0
    else:
        pid = max(sub_dirs_ids)
        if not resume:
            pid+=1
    dir_name += '/' + str(pid)
    if not resume: os.mkdir(dir_name)

    return dir_name    