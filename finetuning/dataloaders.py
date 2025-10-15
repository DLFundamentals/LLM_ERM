import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from itertools import product
import random
import torch.multiprocessing as mp
from sympy import nextprime, isprime
from transformers import AutoTokenizer
import os
import sys
import hashlib
from typing import Callable, Optional
from utils import set_seed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.data_handler import get_data_generator, create_stratified_splits
from src.target_functions import TARGET_FUNCTIONS

random.seed(42)

def collate_sequence(data, targets, collate_option=None):
    
    if collate_option is None:
        return data, targets
    else:
        if collate_option == 'seq':
            return torch.cat([data, targets.unsqueeze(-1)],-1)
        elif collate_option == 'cls':
            return torch.cat([data, torch.full_like(targets, 2).unsqueeze(-1)],-1), targets
        
class CodeDataset:

    def __init__(self, python_code: Callable, sequence_length: int, train_set_size: int, test_set_size: int, batch_size: int, p=0.5, bos_token=2, online=False, device='cpu', dyck2=False, palindrome=False, logger=None, prime=None, pattern=None, prime_odd=None, tokenizer_name=None, cache_dir="finetuning/sgd_datasets_cache", global_seed: int = 42, fn_id: Optional[str] = None, val_set_size: int = 0):
        
        self.python_code = python_code
        self.sequence_length = sequence_length
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.total_size = train_set_size + test_set_size
        self.batch_size = batch_size
        self.device = device
        self.bos_token = bos_token
        self.online = online
        self.logger = logger
        
        # --- Parameters for specific generators ---
        self.dyck2 = dyck2
        self.palindrome = palindrome
        self.prime = prime
        self.pattern = pattern
        self.prime_odd = prime_odd
        
        self.tokenizer = None
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # --- Caching & Determinism Logic (aligned with program_synthesis) ---
        # Prefer the experiment fn_id; fall back to the Python function name for BC.
        fn_key = fn_id if fn_id is not None else self.python_code.__name__
        key = (
            f"{fn_key}|L={self.sequence_length}"
            f"|train={self.train_set_size + val_set_size}"
            f"|test={self.test_set_size}"
            f"|base_seed={global_seed}"
        )
        digest8 = hashlib.sha256(key.encode("utf-8")).digest()[:8]
        derived_seed = (int.from_bytes(digest8, "big") & 0x7FFFFFFF)

        # Cache directory also keyed by this stable string
        self.dataset_cache_dir = os.path.join(
            cache_dir, f"seed_{derived_seed}", hashlib.md5(key.encode()).hexdigest()
        )

        # 2. Define paths for cached tensor files
        train_set_path = os.path.join(self.dataset_cache_dir, 'train_set.pt')
        train_label_path = os.path.join(self.dataset_cache_dir, 'train_label.pt')
        test_set_path = os.path.join(self.dataset_cache_dir, 'test_set.pt')
        test_label_path = os.path.join(self.dataset_cache_dir, 'test_label.pt')
        
        # 3. Check if cached data exists
        if all(os.path.exists(p) for p in [train_set_path, train_label_path, test_set_path, test_label_path]):
            if self.logger:
                self.logger.info(f"Loading cached dataset from: {self.dataset_cache_dir}")
            self.train_set = torch.load(train_set_path, map_location=self.device)
            self.train_label = torch.load(train_label_path, map_location=self.device)
            self.test_set = torch.load(test_set_path, map_location=self.device)
            self.test_label = torch.load(test_label_path, map_location=self.device)
        else:
            # 4. If not, generate it and save it to the cache
            if self.logger:
                self.logger.info(f"No cache found for seed {derived_seed}. Generating dataset...")
            
            # CRITICAL: Reset the seed right before generation
            set_seed(derived_seed)
            if self.logger:
                self.logger.info(f"Random state reset with derived_seed: {derived_seed}")

            self._generate_and_cache_data(train_set_path, train_label_path, test_set_path, test_label_path)

    def _generate_and_cache_data(self, train_set_path, train_label_path, test_set_path, test_label_path):
        """Helper to run generation and save results."""
        # This calls the original data generation logic
        self.generate_datasets_fast()
        
        # Create directory and save tensors
        os.makedirs(self.dataset_cache_dir, exist_ok=True)
        torch.save(self.train_set, train_set_path)
        torch.save(self.train_label, train_label_path)
        torch.save(self.test_set, test_set_path)
        torch.save(self.test_label, test_label_path)
        if self.logger:
            self.logger.info(f"Successfully saved dataset to cache.")

    # In the CodeDataset class

    def generate_datasets_fast(self):
        """
        Generates and splits the dataset by delegating to the centralized data handlers.
        This ensures data generation is consistent across the entire project.
        """
        target_name = None
        for name, func in TARGET_FUNCTIONS.items():
            if func == self.python_code:
                target_name = name
                break
        
        if not target_name:
            if self.palindrome: target_name = "palindrome"
            elif self.pattern == '10101010': target_name = "patternmatch1"
            elif self.pattern == '00111111': target_name = "patternmatch2"
            elif self.dyck2: target_name = "dyck2"
            elif self.prime: target_name = "prime_decimal"
            elif self.prime_odd: target_name = "prime_decimal_tf_check"

        if not target_name:
            raise ValueError(f"Could not determine a target_name for the provided python_code: {self.python_code.__name__}")

        if self.logger:
            self.logger.info(f"Using target_name '{target_name}' for data generation.")
            
        generator = get_data_generator(target_name, self.sequence_length, self.total_size)
        all_samples = generator.generate_data()

        # Use the new centralized splitting function
        train_samples, val_samples, test_samples = create_stratified_splits(
            all_samples=all_samples,
            train_size=self.train_set_size,
            # The finetuning script doesn't have a validation set, so we pass 0.
            # The test set size in the original logic was the remainder.
            val_size=0,
            test_size=self.test_set_size,
            device='cpu'
        )
        # Note: The finetuning in paper had no validation set. Test set was the remainder.
        # We are keeping that logic here. val_samples will be empty.

        # Convert to tensors
        self.train_set = torch.tensor([[int(c) for c in s['Input']] for s in train_samples], dtype=torch.long, device=self.device)
        self.train_label = torch.tensor([int(s['Output']) for s in train_samples], dtype=torch.long, device=self.device)
        self.test_set = torch.tensor([[int(c) for c in s['Input']] for s in test_samples], dtype=torch.long, device=self.device)
        self.test_label = torch.tensor([int(s['Output']) for s in test_samples], dtype=torch.long, device=self.device)

        if self.logger:
            self._validate_split("Train Set", self.train_label)
            self._validate_split("Test Set", self.test_label)

    def _validate_split(self, name, labels):
        """Helper function to log the class distribution of a dataset split."""
        if labels.numel() == 0:
            self.logger.info(f"Validation for '{name}': Dataset is empty.")
            return
            
        num_zeros = (labels == 0).sum().item()
        num_ones = (labels == 1).sum().item()
        total = len(labels)
        self.logger.info(f"Validation for '{name}':")
        self.logger.info(f"  Class 0: {num_zeros} ({num_zeros/total:.2%})")
        self.logger.info(f"  Class 1: {num_ones} ({num_ones/total:.2%})")
        self.logger.info(f"  Total: {total}")
        self.logger.info("-" * 20)

    def create_dataloaders( self, worker=0 ):
        traindata = torch.cat([torch.full((self.train_set.size(0), 1), self.bos_token, device=self.device), self.train_set], dim=1)
        testdata = torch.cat([torch.full((self.test_set.size(0), 1), self.bos_token, device=self.device), self.test_set], dim=1)
        
        train_dataset = TensorDataset(traindata, self.train_label)
        test_dataset = TensorDataset(testdata, self.test_label)
        
        if self.tokenizer:
            # If we are using a pretrained tokenizer, use the custom collator
            from collators import PretrainedDataCollator  # Import here to avoid circular dependency
            collate_fn = PretrainedDataCollator(self.tokenizer, max_length=self.sequence_length + 2)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=worker, collate_fn=collate_fn)
            test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False, num_workers=worker, collate_fn=collate_fn)
        else:
            # From-scratch models: default collate returns (inputs, labels) tensors
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=worker)
            test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False, num_workers=worker)

        return train_loader, test_loader