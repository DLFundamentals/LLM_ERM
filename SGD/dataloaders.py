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
import hashlib
from typing import Callable
from utils import set_seed

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

    def __init__(self, python_code: Callable, sequence_length: int, train_set_size: int, test_set_size: int, batch_size: int, p=0.5, bos_token=2, online=False, device='cpu', dyck2=False, palindrome=False, logger=None, prime=None, pattern=None, prime_odd=None, tokenizer_name=None, cache_dir="sgd_datasets_cache", global_seed: int = 42):
        
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

        # --- New Caching & Determinism Logic ---
        # 1. Create a unique identifier based on the GLOBAL seed and config
        config_str = (
            f"func={self.python_code.__name__}-len={self.sequence_length}-"
            f"train={self.train_set_size}-test={self.test_set_size}"
        )
        # Create a unique derived seed for this specific task configuration
        h = hashlib.sha256(config_str.encode()).digest()
        derived_seed = (int.from_bytes(h, 'big') & 0x7fffffff) ^ global_seed
        
        # The cache path now depends on the derived seed, ensuring no collisions if global_seed changes
        self.dataset_cache_dir = os.path.join(cache_dir, f"seed_{derived_seed}", hashlib.md5(config_str.encode()).hexdigest())

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

    def _generate_pattern_match(self, pattern_received):
        """
        Generate 50% positives (contain pattern) and 50% negatives (guaranteed NOT to contain it).
        All heavy work happens on CPU to avoid VRAM spikes; move to self.device at the end.
        """
        half_size = self.total_size // 2
        if pattern_received == '10101010':
            pattern_bits = [1, 0, 1, 0, 1, 0, 1, 0]
        elif pattern_received == '00111111':
            pattern_bits = [0, 0, 1, 1, 1, 1, 1, 1]
        else:
            raise Exception(f"pattern match data creation not implemented for {pattern_received}")
        pattern_len = len(pattern_bits)

        assert self.total_size % 2 == 0, "Total size must be even for pattern generation"
        assert self.sequence_length >= pattern_len, f"Sequence length must be at least {pattern_len}"

        # ---- Positives (contains the pattern) on CPU ----
        pos = torch.randint(0, 2, (half_size, self.sequence_length), dtype=torch.long, device="cpu")
        p_cpu = torch.tensor(pattern_bits, dtype=torch.long, device="cpu")
        starts = torch.randint(0, self.sequence_length - pattern_len + 1, (half_size,), device="cpu")
        # insert pattern per row
        for i in range(half_size):
            s = starts[i].item()
            pos[i, s:s + pattern_len] = p_cpu

        # ---- Negatives (guaranteed no pattern) on CPU, in batches ----
        neg_chunks = []
        needed = half_size
        # pick a decent batch size; increase if you want fewer loops
        BATCH = max(8192, min(65536, half_size * 2))
        while needed > 0:
            cand = torch.randint(0, 2, (BATCH, self.sequence_length), dtype=torch.long, device="cpu")
            windows = cand.unfold(dimension=1, size=pattern_len, step=1)  # (BATCH, L - k + 1, k)
            has_pat = torch.all(windows == p_cpu, dim=2).any(dim=1)       # (BATCH,)
            valid = cand[~has_pat]
            if valid.numel() > 0:
                neg_chunks.append(valid)
                needed -= valid.size(0)
        neg = torch.cat(neg_chunks, dim=0)[:half_size]

        # ---- Concatenate and return on requested device ----
        binary_samples = torch.cat([pos, neg], dim=0).to(self.device, non_blocking=False)
        return binary_samples

    def _generate_primes_odd(self):
        """
        Generates a dataset with 50% primes and 50% non-primes of a specific digit length.
        Uses sympy for robust primality testing and constructs non-primes for efficiency.
        """
        if self.total_size % 2 != 0:
            raise ValueError("Total size must be even for a 50/50 prime/non-prime split.")

        num_primes_needed = self.total_size // 2
        num_non_primes_needed = self.total_size // 2

        start_range = 10**(self.sequence_length - 1)
        end_range = 10**self.sequence_length

        # Validate if the request is possible
        max_possible = end_range - start_range
        if self.total_size > max_possible:
            raise ValueError(f"Requested {self.total_size} samples, but only {max_possible} unique numbers of length {self.sequence_length} exist.")

        if self.logger:
            self.logger.info(f"Generating {num_primes_needed} primes and {num_non_primes_needed} non-primes...")

        primes_found = set()
        non_primes_found = set()

        # --- Generate Primes ---
        while len(primes_found) < num_primes_needed:
            # Pick a random starting point in the range
            random_start = random.randint(start_range, end_range - 1)
            # Find the next prime number from that point
            candidate = nextprime(random_start)
            
            # If the found prime is still within our desired digit length and not already found
            if (candidate < end_range) and (candidate not in primes_found):
                primes_found.add(candidate)

        # --- Generate Non-Primes (by construction) ---
        # For odd L, use floor(L/2)-digit × ceil(L/2)-digit so product often lands in L digits.
        a_len = self.sequence_length // 2                 # floor(L/2)
        b_len = self.sequence_length - a_len              # ceil(L/2)
        a_start, a_end = 10**(a_len - 1), 10**a_len       # [a_start, a_end)
        b_start, b_end = 10**b_len // 10, 10**b_len       # same as 10**(b_len - 1), written safely

        all_found = primes_found.copy()  # Avoid collisions with primes

        while len(non_primes_found) < num_non_primes_needed:
            f1 = random.randrange(a_start, a_end)
            f2 = random.randrange(b_start, b_end)
            candidate = f1 * f2

            # Keep only exact-length composites and ensure uniqueness
            if start_range <= candidate < end_range and candidate not in all_found and (candidate%10 not in [0, 2, 4, 5, 6, 8]):
                non_primes_found.add(candidate)
                all_found.add(candidate)
                
        # --- Convert numbers to tensors of digits ---
        all_numbers = list(primes_found) + list(non_primes_found)
        
        digit_sequences = [[int(digit) for digit in str(num)] for num in all_numbers]
        data_tensor = torch.tensor(digit_sequences, dtype=torch.long, device=self.device)
        return data_tensor

    def _generate_primes(self):
        """
        Generates a dataset with 50% primes and 50% non-primes of a specific digit length.
        Uses sympy for robust primality testing and constructs non-primes for efficiency.
        """
        if self.total_size % 2 != 0:
            raise ValueError("Total size must be even for a 50/50 prime/non-prime split.")

        num_primes_needed = self.total_size // 2
        num_non_primes_needed = self.total_size // 2

        start_range = 10**(self.sequence_length - 1)
        end_range = 10**self.sequence_length

        # Validate if the request is possible
        max_possible = end_range - start_range
        if self.total_size > max_possible:
            raise ValueError(f"Requested {self.total_size} samples, but only {max_possible} unique numbers of length {self.sequence_length} exist.")

        if self.logger:
            self.logger.info(f"Generating {num_primes_needed} primes and {num_non_primes_needed} non-primes...")

        primes_found = set()
        non_primes_found = set()

        # --- Generate Primes ---
        while len(primes_found) < num_primes_needed:
            # Pick a random starting point in the range
            random_start = random.randint(start_range, end_range - 1)
            # Find the next prime number from that point
            candidate = nextprime(random_start)
            
            # If the found prime is still within our desired digit length and not already found
            if candidate < end_range and candidate not in primes_found:
                primes_found.add(candidate)

        # --- Generate Non-Primes (by construction) ---
        # For odd L, use floor(L/2)-digit × ceil(L/2)-digit so product often lands in L digits.
        a_len = self.sequence_length // 2                 # floor(L/2)
        b_len = self.sequence_length - a_len              # ceil(L/2)
        a_start, a_end = 10**(a_len - 1), 10**a_len       # [a_start, a_end)
        b_start, b_end = 10**b_len // 10, 10**b_len       # same as 10**(b_len - 1), written safely

        all_found = primes_found.copy()  # Avoid collisions with primes

        while len(non_primes_found) < num_non_primes_needed:
            f1 = random.randrange(a_start, a_end)
            f2 = random.randrange(b_start, b_end)
            candidate = f1 * f2

            # Keep only exact-length composites and ensure uniqueness
            if start_range <= candidate < end_range and candidate not in all_found:
                non_primes_found.add(candidate)
                all_found.add(candidate)
                
        # --- Convert numbers to tensors of digits ---
        all_numbers = list(primes_found) + list(non_primes_found)
        
        digit_sequences = [[int(digit) for digit in str(num)] for num in all_numbers]
        data_tensor = torch.tensor(digit_sequences, dtype=torch.long, device=self.device)
        return data_tensor

    def _generate_palindromes(self):
        """Generates a dataset with 50% palindromes and 50% non-palindromes."""
        half_size = self.total_size // 2
        
        # Ensure total_size is even for a perfect 50/50 split
        assert self.total_size % 2 == 0, "Total size must be even for palindrome generation"

        # 1. Generate Palindromes
        # A palindrome is defined by its first half.
        # Length of the first half, handling both odd and even sequence lengths
        half_len = (self.sequence_length + 1) // 2
        
        # Generate random first halves
        first_halves = torch.randint(0, 2, size=(half_size, half_len), device=self.device, dtype=torch.long)
        
        # Create the second half by reversing the first part (excluding the middle element if odd)
        second_halves_reversed = torch.flip(first_halves[:, :self.sequence_length // 2], dims=[1])
        
        # Concatenate to form full palindromes
        palindromes = torch.cat([first_halves, second_halves_reversed], dim=1)

        # 2. Generate Non-Palindromes
        # An easy way to guarantee a non-palindrome is to create a palindrome and flip a single bit in the first half.
        # This ensures the symmetry is broken.
        non_palindromes = palindromes.clone() # Start with the same palindromes
        
        # Pick a random index in the first half of each sequence to flip
        row_indices = torch.arange(half_size, device=self.device)
        col_indices_to_flip = torch.randint(0, half_len, size=(half_size,), device=self.device)
        
        # Flip the bits at the chosen locations (0 becomes 1, 1 becomes 0)
        non_palindromes[row_indices, col_indices_to_flip] = 1 - non_palindromes[row_indices, col_indices_to_flip]
        
        # 3. Combine and return
        binary_samples = torch.cat([palindromes, non_palindromes], dim=0)
        
        return binary_samples

    def _generate_balanced_binary(self):
        assert self.total_size % 2 == 0, "Total size must be even."
        target_per_class = self.total_size // 2

        # Work entirely on CPU to avoid VRAM spikes; cast to long only when needed
        BATCH = max(8192, min(131072, self.total_size * 2))
        buf0, buf1 = [], []
        n0 = n1 = 0

        # Helper: label on CPU if possible; otherwise minimally on GPU
        def label_batch(x_cpu):
            return self.python_code(x_cpu.to(torch.long), device="cpu").to(torch.long)
            # If they MUST run on CUDA, use:
            # x_gpu = x_cpu.to("cuda", non_blocking=True, dtype=torch.long)
            # y = self.python_code(x_gpu, device="cuda").to(torch.long).cpu()
            # del x_gpu
            # return y

        attempts = 0
        max_attempts = 10000  # generous but finite
        while (n0 < target_per_class or n1 < target_per_class) and attempts < max_attempts:
            cand = torch.randint(0, 2, (BATCH, self.sequence_length), dtype=torch.uint8, device="cpu")
            y = label_batch(cand)

            idx0 = (y == 0).nonzero(as_tuple=True)[0]
            idx1 = (y == 1).nonzero(as_tuple=True)[0]

            need0 = target_per_class - n0
            need1 = target_per_class - n1
            if need0 > 0 and idx0.numel() > 0:
                take0 = idx0[:need0]
                buf0.append(cand.index_select(0, take0))
                n0 += take0.numel()
            if need1 > 0 and idx1.numel() > 0:
                take1 = idx1[:need1]
                buf1.append(cand.index_select(0, take1))
                n1 += take1.numel()

            attempts += 1
            # Optional early warning for extreme imbalance
            if attempts % 50 == 0:
                r0 = (idx0.numel() / BATCH)
                r1 = (idx1.numel() / BATCH)
                if min(r0, r1) < 0.01:
                    raise RuntimeError(f"Label imbalance too extreme (r0={r0:.4f}, r1={r1:.4f}). "
                                    f"Consider smaller target_per_class or different generator.")

        if attempts >= max_attempts:
            raise RuntimeError(f"Failed to balance after {max_attempts} attempts: got {n0} zeros, {n1} ones.")

        data0 = torch.cat(buf0, dim=0)[:target_per_class].to(torch.long)
        data1 = torch.cat(buf1, dim=0)[:target_per_class].to(torch.long)
        binary_samples_cpu = torch.cat([data0, data1], dim=0)

        # Move final tensor to requested device only once
        return binary_samples_cpu.to(self.device, non_blocking=False)


    def generate_datasets_fast( self ):
        
        def generate_dyck2():
            assert self.sequence_length % 4 == 0
            assert self.total_size % 2 == 0

            paren_map = {"00": "(", "01": ")", "10": "[", "11": "]"}
            paren_to_bit = {v: k for k, v in paren_map.items()}
            open_to_close = {'(': ')', '[': ']'}

            def to_bits(seq):
                return ''.join(paren_to_bit[c] for c in seq)

            def generate_valid_seq():
                stack, seq = [], []
                while (len(stack) + len(seq)) < (self.sequence_length//2):
                    o, c = random.choice(list(open_to_close.items()))
                    if stack:
                        if random.random() < 0.5:
                            seq.append( stack.pop() )
                        else:
                            seq.append( o )
                            stack.append( c )
                    else:
                        seq.append( o )
                        stack.append( c )
                while len(stack) > 0:
                    seq.append( stack.pop() )
                return seq + stack
            
            def generate_random_seq( seq ):
                l = list(seq)
                
                k = random.choice([1, 2, 3, 4, 5])  # Number of bits to flip
                positions = random.sample(range(len(l)), k)  # Choose k distinct positions
                
                for pos in positions:
                    l[pos] = '1' if l[pos] == '0' else '0'  # Flip bit
                
                return ''.join(l)

            # Generate balanced sets of valid and invalid sequences
            valid_set = set()
            while len(valid_set) < self.total_size // 2:
                valid_set.add(to_bits(generate_valid_seq()))

            invalid_set = set()
            while len(invalid_set) < self.total_size // 2:
                invalid_set.add(generate_random_seq(to_bits(generate_valid_seq())))
            
            # Convert to tensors
            valid_ints = [torch.tensor([int(ch) for ch in s], dtype=torch.long, device=self.device) for s in valid_set]
            valid_ints = torch.stack(valid_ints, dim=0)
            invalid_ints = [torch.tensor([int(ch) for ch in s], dtype=torch.long, device=self.device) for s in invalid_set]
            invalid_ints = torch.stack(invalid_ints, dim=0)

            return torch.cat((valid_ints, invalid_ints), dim=0)

        # === STEP 1: Generate a balanced source dataset ===
        if self.dyck2:
            binary_samples = generate_dyck2()
        elif self.palindrome:
            binary_samples = self._generate_palindromes()
        elif self.prime:
            binary_samples = self._generate_primes()
        elif self.prime_odd:
            binary_samples = self._generate_primes_odd()
        elif self.pattern:
            binary_samples = self._generate_pattern_match( self.pattern )
        else:
            # For the generic case, we must generate-and-label to ensure balance
            binary_samples = self._generate_balanced_binary()
        
        # === STEP 2: Perform a stratified split to ensure train/test sets are also 50/50 ===
        
        # First, calculate labels for the entire generated dataset
        all_labels = self.python_code(binary_samples, self.device).long()
        
        # Get indices for each class
        indices_0 = torch.where(all_labels == 0)[0]
        indices_1 = torch.where(all_labels == 1)[0]
        
        # Check if we have enough samples of each class for the split
        assert self.train_set_size % 2 == 0, "train_set_size must be even for a balanced split."
        num_train_per_class = self.train_set_size // 2
        
        if len(indices_0) < num_train_per_class or len(indices_1) < num_train_per_class:
            raise ValueError(f"Not enough samples to create a balanced training set. "
                             f"Required {num_train_per_class} per class, but got {len(indices_0)} of class 0 "
                             f"and {len(indices_1)} of class 1.")
        
        # Shuffle indices within each class to ensure random sampling
        shuffled_indices_0 = indices_0[torch.randperm(len(indices_0))]
        shuffled_indices_1 = indices_1[torch.randperm(len(indices_1))]

        # Select an equal number of samples from each class for the training set
        train_indices = torch.cat([
            shuffled_indices_0[:num_train_per_class],
            shuffled_indices_1[:num_train_per_class]
        ])
        
        # Use the remaining indices for the test set
        test_indices = torch.cat([
            shuffled_indices_0[num_train_per_class:], 
            shuffled_indices_1[num_train_per_class:]
        ])

        # Create final datasets, shuffling the training set order so batches are mixed
        train_indices_shuffled = train_indices[torch.randperm(len(train_indices))]
        
        self.train_set = binary_samples[train_indices_shuffled]
        self.train_label = all_labels[train_indices_shuffled]
        self.test_set = binary_samples[test_indices]
        self.test_label = all_labels[test_indices]

        # Optional: Validation logging
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
        
        if self.tokenizer:
            # If we are using a pretrained tokenizer, use the custom collator
            from collators import PretrainedDataCollator # Import here to avoid circular dependency
            collate_fn = PretrainedDataCollator(self.tokenizer, max_length=self.sequence_length + 2) # A bit of buffer
            
            train_loader = DataLoader(list(zip(traindata, self.train_label)), batch_size=self.batch_size, shuffle=True, num_workers=worker, collate_fn=collate_fn)
            test_loader = DataLoader(list(zip(testdata, self.test_label)), batch_size=self.batch_size, shuffle=False, num_workers=worker, collate_fn=collate_fn)
        else:
            # Original behavior for from-scratch models
            train_loader = DataLoader(list(zip(traindata, self.train_label)), batch_size=self.batch_size, shuffle=True, num_workers=worker)
            test_loader = DataLoader(list(zip(testdata, self.test_label)), batch_size=self.batch_size, shuffle=False, num_workers=worker)

        return train_loader, test_loader