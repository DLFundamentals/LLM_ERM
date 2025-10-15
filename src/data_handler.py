# data_handler.py
"""
Data generation module for various machine learning tasks.

This module provides a collection of data generator classes built on a common
abstract base class, `BaseDataGenerator`. Each generator is responsible for
creating a balanced (50/50 split) dataset for a specific task, such as
primality testing, palindrome detection, or formal language recognition.

Available Generators:
- PrimeDecimalTailRestrictedDataGenerator: Prime vs. composite ending in {1,3,7,9}.
- PrimeDataGenerator: Slower, sympy-based prime number data generation.
- BinaryDataGenerator: Generic generator for various binary target functions.
- Dyck2DataGenerator: Generates sequences for the Dyck-2 language (e.g., '()[]').
- PatternBasedDataGenerator: Detects a specific binary pattern in sequences.
- PalindromeDataGenerator: Detects if a binary sequence is a palindrome.
"""

import torch
import math
import random
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Set, Tuple, Union
from abc import ABC, abstractmethod

from .target_functions import TARGET_FUNCTIONS

# --- Module-level Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional dependency handling for performance
try:
    from sympy import nextprime, isprime
except ImportError:
    logger.error("Error: sympy library not found. Please run 'pip install sympy'")
    exit()


# =============================================================================
# Abstract Base Class for All Data Generators
# =============================================================================

class BaseDataGenerator(ABC):
    """
    Abstract base class for data generators.

    This class provides a common structure for all data generators, handling
    initialization, and validation.
    Subclasses are required to implement the core logic for generating raw data
    and formatting individual samples.
    """

    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            msg = f"Sequence length must be a positive integer, but got {sequence_length}."
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(num_samples, int) or num_samples <= 0:
            msg = f"Number of samples must be a positive integer, but got {num_samples}."
            logger.error(msg)
            raise ValueError(msg)

        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.device = device

        if num_samples % 2 != 0:
            logger.warning(
                f"{self.__class__.__name__}: num_samples is odd ({num_samples}). "
                "For a 50/50 split, an even number is recommended."
            )
        self.num_positive_samples = self.num_samples // 2
        self.num_negative_samples = self.num_samples - self.num_positive_samples

    @abstractmethod
    def _generate_raw_data(self) -> Tuple[List[Any], List[Any]]:
        """
        Generates the raw data for positive and negative samples.
        Must be implemented by subclasses.

        Returns:
            A tuple of two lists: (positive_samples, negative_samples). The elements
            of the lists can be any type (e.g., int, torch.Tensor) that the
            _format_input method can handle.
        """
        pass

    @abstractmethod
    def _format_input(self, sample: Any) -> np.ndarray:
        """
        Formats a single raw sample into the required numpy array of strings.
        Must be implemented by subclasses.
        """
        pass

    def generate_data(self) -> List[Dict[str, Any]]:
        """
        Orchestrates the data generation, formatting, and shuffling process.
        This is the main public method to be called by users.
        """
        class_name = self.__class__.__name__
        logger.info(f"Starting data generation for {class_name}...")

        positive_samples, negative_samples = self._generate_raw_data()

        # Defensive checks to ensure subclass implementation is correct
        if len(positive_samples) != self.num_positive_samples:
            raise RuntimeError(f"{class_name} generated {len(positive_samples)} positive samples, expected {self.num_positive_samples}.")
        if len(negative_samples) != self.num_negative_samples:
            raise RuntimeError(f"{class_name} generated {len(negative_samples)} negative samples, expected {self.num_negative_samples}.")

        logger.info("Formatting and combining dataset...")
        dataset = []
        for p_sample in tqdm(positive_samples, desc="Formatting positive samples", leave=False, unit="sample"):
            dataset.append({'Input': self._format_input(p_sample), 'Output': '1'})

        for n_sample in tqdm(negative_samples, desc="Formatting negative samples", leave=False, unit="sample"):
            dataset.append({'Input': self._format_input(n_sample), 'Output': '0'})

        logger.info(f"Data generation complete for {class_name}. Total samples: {len(dataset)}.")
        return dataset


# =============================================================================
# Primality-Based Data Generators
# =============================================================================


class BaseDecimalGenerator(BaseDataGenerator):
    """Base class for decimal-based generators that support leading zeros."""
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu', allow_leading_zeros: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if self.num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a guaranteed 50/50 split.")

        self.allow_leading_zeros = allow_leading_zeros
        if allow_leading_zeros:
            self.start_range = 0
            self.end_range = 10 ** self.sequence_length
        else:
            self.start_range = 10 ** (self.sequence_length - 1) if self.sequence_length > 1 else 1
            self.end_range = 10 ** self.sequence_length

        max_possible = self.end_range - self.start_range
        if self.num_samples > max_possible:
            raise ValueError(f"Requested {self.num_samples} samples, but only {max_possible} unique sequences exist.")

    def _format_input(self, sample: int) -> np.ndarray:
        return np.array(list(str(sample).zfill(self.sequence_length)))
    

class PrimeDecimalTailRestrictedDataGenerator(BaseDecimalGenerator):
    """Generates primes vs. non-primes ending in a decimal from {1, 3, 7, 9}."""

    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu', allow_leading_zeros: bool = False,
                 allowed_nonprime_last_digits: Tuple[int, ...] = (1, 3, 7, 9)):
        super().__init__(sequence_length, num_samples, device, allow_leading_zeros)
        if not all(d in range(10) for d in allowed_nonprime_last_digits):
            raise ValueError("allowed_nonprime_last_digits must be decimal digits 0-9.")
        self.allowed_nonprime_last_digits = tuple(sorted(set(allowed_nonprime_last_digits)))
        logger.info(f"PrimeDecimalTailRestrictedDataGenerator initialized with allowed_nonprime_last_digits={self.allowed_nonprime_last_digits}")

    def _generate_raw_data(self) -> Tuple[List[int], List[int]]:
        primes_found: Set[int] = set()
        non_primes_found: Set[int] = set()

        # --- Generate Primes ---
        while len(primes_found) < self.num_positive_samples:
            rnd_start = random.randint(self.start_range, max(self.start_range, self.end_range - 2))
            candidate = nextprime(rnd_start)
            if self.start_range <= candidate < self.end_range and candidate not in primes_found:
                primes_found.add(candidate)

        # --- Generate Non-Primes (CONSTRUCTIVE method + last digit filter) ---
        a_len = self.sequence_length // 2
        b_len = self.sequence_length - a_len
        a_start, a_end = 10**(a_len - 1), 10**a_len
        b_start, b_end = 10**(b_len - 1), 10**b_len

        all_found = primes_found.copy()

        while len(non_primes_found) < self.num_negative_samples:
            f1 = random.randrange(a_start, a_end) if a_start < a_end else a_start
            f2 = random.randrange(b_start, b_end) if b_start < b_end else b_start
            candidate = f1 * f2
            
            # Keep only exact-length composites with the correct last digit
            if (self.start_range <= candidate < self.end_range and 
                candidate not in all_found and 
                (candidate % 10) in self.allowed_nonprime_last_digits):
                non_primes_found.add(candidate)
                all_found.add(candidate)
        
        logger.info(f"Generated {len(primes_found)} primes and {len(non_primes_found)} restricted non-primes.")
        return list(primes_found), list(non_primes_found)
    

class PrimeDataGenerator(BaseDataGenerator):
    """Generates decimal input vectors and corresponding prime/non-prime outputs using sympy."""
    
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        super().__init__(sequence_length, num_samples, device)
        self.start_range = 10**(self.sequence_length - 1) if self.sequence_length > 1 else 1
        self.end_range = 10**self.sequence_length
        if self.num_samples > (self.end_range - self.start_range):
            raise ValueError(f"Requested {self.num_samples} samples, but only {self.end_range - self.start_range} unique numbers exist.")
        logger.info(f"PrimeDataGenerator (sympy) initialized for len={sequence_length}, samples={num_samples}")

    def _format_input(self, sample: int) -> np.ndarray:
        return np.array(list(str(sample)))

    def _generate_raw_data(self) -> Tuple[List[int], List[int]]:
        primes_found: Set[int] = set()
        non_primes_found: Set[int] = set()

        # --- Generate Primes ---
        while len(primes_found) < self.num_positive_samples:
            random_start = random.randint(self.start_range, self.end_range - 2) if self.end_range - 2 > self.start_range else self.start_range
            candidate = nextprime(random_start)
            if candidate < self.end_range and candidate not in primes_found:
                primes_found.add(candidate)
        
        # --- Generate Non-Primes (CONSTRUCTIVE method from SGD script) ---
        a_len = self.sequence_length // 2
        b_len = self.sequence_length - a_len
        a_start, a_end = 10**(a_len - 1), 10**a_len
        b_start, b_end = 10**(b_len - 1), 10**b_len

        all_found = primes_found.copy() # Avoid generating a number that is already in the prime set

        while len(non_primes_found) < self.num_negative_samples:
            f1 = random.randrange(a_start, a_end) if a_start < a_end else a_start
            f2 = random.randrange(b_start, b_end) if b_start < b_end else b_start
            candidate = f1 * f2

            # Keep only exact-length composites and ensure uniqueness
            if self.start_range <= candidate < self.end_range and candidate not in all_found:
                non_primes_found.add(candidate)
                all_found.add(candidate)

        logger.info(f"Successfully generated {len(primes_found)} primes and {len(non_primes_found)} non-primes.")
        return list(primes_found), list(non_primes_found)


# =============================================================================
# Tensor-Based Data Generators
# =============================================================================

class BaseTensorGenerator(BaseDataGenerator):
    """Base class for generators that internally use torch.Tensors."""
    def _format_input(self, sample: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        return np.array([str(item.item()) for item in sample])


class BinaryDataGenerator(BaseTensorGenerator):
    """Generates binary vectors from a target function with an exact 50/50 split."""
    _MAX_ATTEMPTS_NO_PROGRESS = 50

    def __init__(self, target_function_name: str, sequence_length: int, num_samples: int,
                 device: str = 'cpu', allow_duplicates: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        if target_function_name not in TARGET_FUNCTIONS:
            raise ValueError(f"Unknown target function: {target_function_name}.")
        self.target_function = TARGET_FUNCTIONS[target_function_name]
        self.allow_duplicates = allow_duplicates
        if not allow_duplicates and num_samples > 2**sequence_length:
            raise ValueError(f"Cannot generate {num_samples} unique samples; only {2**sequence_length} are possible.")
        logger.info(f"BinaryDataGenerator initialized for target='{target_function_name}', allow_duplicates={allow_duplicates}")

    def _generate_raw_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        samples_1, samples_0 = self._generate_balanced_samples()
        return list(samples_1), list(samples_0)

    def _generate_balanced_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.allow_duplicates:
            return self._generate_with_duplicates()
        return self._generate_unique()

    def _generate_with_duplicates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        samples_0, samples_1 = [], []
        attempts = 0
        while (len(samples_0) < self.num_positive_samples) or (len(samples_1) < self.num_negative_samples):
            need0 = self.num_positive_samples - len(samples_0)
            need1 = self.num_negative_samples - len(samples_1)
            batch_size = int(max(need0, need1, 1000) * 1.5)
            candidates = torch.randint(0, 2, (batch_size, self.sequence_length), dtype=torch.long, device=self.device)
            outputs = self.target_function(candidates, self.device)

            idx0 = (outputs == 0).nonzero(as_tuple=True)[0]
            idx1 = (outputs == 1).nonzero(as_tuple=True)[0]
            progress = idx0.numel() > 0 or idx1.numel() > 0

            if need0 > 0 and idx0.numel() > 0: samples_0.extend(candidates[idx0])
            if need1 > 0 and idx1.numel() > 0: samples_1.extend(candidates[idx1])
            
            attempts = 0 if progress else attempts + 1
            if attempts > self._MAX_ATTEMPTS_NO_PROGRESS:
                raise RuntimeError("Failed to gather samples. Target function may be too skewed.")

        return torch.stack(samples_1[:self.num_positive_samples]), torch.stack(samples_0[:self.num_negative_samples])

    def _generate_unique(self) -> Tuple[torch.Tensor, torch.Tensor]:
        found_set: Set[Tuple[int, ...]] = set()
        samples_0, samples_1 = [], []
        attempts = 0
        while (len(samples_0) < self.num_positive_samples) or (len(samples_1) < self.num_negative_samples):
            needed = (self.num_positive_samples - len(samples_0)) + (self.num_negative_samples - len(samples_1))
            batch_size = int(max(needed, 1000) * 1.5)
            candidates = torch.randint(0, 2, (batch_size, self.sequence_length), dtype=torch.long, device=self.device)
            outputs = self.target_function(candidates, self.device)

            progress = False
            for cand, out in zip(candidates, outputs):
                cand_tuple = tuple(cand.tolist())
                if cand_tuple in found_set: continue
                if out == 0 and len(samples_0) < self.num_negative_samples:
                    samples_0.append(cand)
                    found_set.add(cand_tuple)
                    progress = True
                elif out == 1 and len(samples_1) < self.num_positive_samples:
                    samples_1.append(cand)
                    found_set.add(cand_tuple)
                    progress = True
            
            attempts = 0 if progress else attempts + 1
            if attempts > self._MAX_ATTEMPTS_NO_PROGRESS:
                raise RuntimeError("Failed to gather unique samples. Target function may be too skewed.")

        return torch.stack(samples_1), torch.stack(samples_0)


class Dyck2DataGenerator(BaseTensorGenerator):
    """Generates sequences for the Dyck-2 language (e.g., '()[]') with a 50/50 split."""
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu', allow_duplicates: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if sequence_length <= 0 or sequence_length % 4 != 0:
            raise ValueError("sequence_length must be a positive multiple of 4.")
        if num_samples % 2 != 0: raise ValueError("num_samples must be even.")
        self.paren_seq_length = sequence_length // 2
        self.paren_map = {"00": "(", "01": ")", "10": "[", "11": "]"}
        self.paren_to_bit_str = {v: k for k, v in self.paren_map.items()}
        self.open_to_close = {'(': ')', '[': ']'}
        self.close_to_open = {v: k for k, v in self.open_to_close.items()}
        self.allow_duplicates = allow_duplicates
        logger.info(f"Dyck2DataGenerator initialized, allow_duplicates={allow_duplicates}")

    def _generate_raw_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.allow_duplicates:
            valid_samples = self._generate_valid_samples_with_duplicates(self.num_positive_samples)
            invalid_samples = self._generate_invalid_samples_with_duplicates(self.num_negative_samples)
        else:
            valid_samples_set = self._generate_valid_samples_unique(self.num_positive_samples)
            invalid_samples_set = self._generate_invalid_samples_unique(self.num_negative_samples, valid_samples_set)
            valid_samples = torch.tensor(list(valid_samples_set), dtype=torch.long, device=self.device)
            invalid_samples = torch.tensor(list(invalid_samples_set), dtype=torch.long, device=self.device)

        return list(valid_samples), list(invalid_samples)

    def _is_valid_paren_seq(self, paren_seq: str) -> bool:
        stack = []
        for char in paren_seq:
            if char in self.open_to_close:
                stack.append(char)
            elif char in self.close_to_open:
                if not stack or stack.pop() != self.close_to_open[char]:
                    return False
        return not stack

    def _generate_one_valid_paren_seq(self) -> str:
        stack, seq = [], []
        while (len(stack) + len(seq)) < self.paren_seq_length:
            o, c = random.choice(list(self.open_to_close.items()))
            if not stack or random.random() < 0.5:
                seq.append(o)
                stack.append(c)
            else:
                seq.append(stack.pop())
        while stack:
            seq.append(stack.pop())
        return "".join(seq)

    def _generate_valid_samples_with_duplicates(self, count: int) -> torch.Tensor:
        samples = []
        for _ in range(count):
            paren_seq = self._generate_one_valid_paren_seq()
            bit_string = "".join(self.paren_to_bit_str[c] for c in paren_seq)
            samples.append(torch.tensor([int(b) for b in bit_string], dtype=torch.long))
        return torch.stack(samples, dim=0).to(self.device)

    def _generate_invalid_samples_with_duplicates(self, count: int) -> torch.Tensor:
        samples = []
        while len(samples) < count:
            valid_paren_seq = self._generate_one_valid_paren_seq()
            bits = list(''.join(self.paren_to_bit_str[c] for c in valid_paren_seq))
            # Corrupt 1 to 5 bits to create an invalid sequence
            num_flips = min(random.randint(1, 5), self.sequence_length)
            for pos in random.sample(range(self.sequence_length), k=num_flips):
                bits[pos] = '1' if bits[pos] == '0' else '0'
            corrupted_bit_str = "".join(bits)
            bit_pairs = [corrupted_bit_str[i:i+2] for i in range(0, self.sequence_length, 2)]

            if all(pair in self.paren_map for pair in bit_pairs):
                paren_seq = "".join([self.paren_map[p] for p in bit_pairs])
                if not self._is_valid_paren_seq(paren_seq):
                    samples.append(torch.tensor([int(b) for b in corrupted_bit_str], dtype=torch.long))
            else: # Guaranteed invalid if bit pairs are not well-formed
                samples.append(torch.tensor([int(b) for b in corrupted_bit_str], dtype=torch.long))
        return torch.stack(samples, dim=0).to(self.device)

    def _generate_valid_samples_unique(self, count: int) -> Set[Tuple[int, ...]]:
        unique_samples: Set[Tuple[int, ...]] = set()
        while len(unique_samples) < count:
            paren_seq = self._generate_one_valid_paren_seq()
            bit_string = "".join(self.paren_to_bit_str[c] for c in paren_seq)
            bit_tuple = tuple(int(b) for b in bit_string)
            unique_samples.add(bit_tuple)
        return unique_samples

    def _generate_invalid_samples_unique(self, count: int, existing_samples: Set[Tuple[int, ...]]) -> Set[Tuple[int, ...]]:
        unique_invalid: Set[Tuple[int, ...]] = set()
        while len(unique_invalid) < count:
            valid_paren_seq = self._generate_one_valid_paren_seq()
            bits = list(''.join(self.paren_to_bit_str[c] for c in valid_paren_seq))
            num_flips = min(random.randint(1, 5), self.sequence_length)
            for pos in random.sample(range(self.sequence_length), k=num_flips):
                bits[pos] = '1' if bits[pos] == '0' else '0'
            corrupted_bit_str = "".join(bits)
            
            is_valid_after_corruption = False
            bit_pairs = [corrupted_bit_str[i:i+2] for i in range(0, self.sequence_length, 2)]
            if all(pair in self.paren_map for pair in bit_pairs):
                paren_seq = "".join([self.paren_map[p] for p in bit_pairs])
                if self._is_valid_paren_seq(paren_seq):
                    is_valid_after_corruption = True

            if not is_valid_after_corruption:
                candidate_tuple = tuple(int(b) for b in corrupted_bit_str)
                if candidate_tuple not in existing_samples and candidate_tuple not in unique_invalid:
                    unique_invalid.add(candidate_tuple)
        return unique_invalid


class PatternBasedDataGenerator(BaseTensorGenerator):
    """Generates a 50/50 dataset of binary sequences with/without a given pattern."""
    def __init__(self, sequence_length: int, num_samples: int, pattern_string: str = '10101010',
                 device: str = 'cpu', allow_duplicates: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0: raise ValueError("num_samples must be even.")
        if not pattern_string or len(pattern_string) > sequence_length:
            raise ValueError("Invalid pattern_string length.")
        try:
            self.pattern_tensor = torch.tensor([int(b) for b in pattern_string], dtype=torch.long, device=device)
        except ValueError:
            raise ValueError("pattern_string must contain only '0's and '1's.")
        self.allow_duplicates = allow_duplicates
        logger.info(f"PatternBasedDataGenerator initialized for pattern='{pattern_string}', allow_duplicates={allow_duplicates}")

    def _generate_raw_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.allow_duplicates:
            with_p = self._generate_with_pattern_duplicates(self.num_positive_samples)
            without_p = self._generate_without_pattern_duplicates(self.num_negative_samples)
        else:
            with_p_set = self._generate_with_pattern_unique(self.num_positive_samples)
            without_p_set = self._generate_without_pattern_unique(self.num_negative_samples, with_p_set)
            with_p = torch.tensor(list(with_p_set), dtype=torch.long, device=self.device)
            without_p = torch.tensor(list(without_p_set), dtype=torch.long, device=self.device)
        return list(with_p), list(without_p)

    def _generate_with_pattern_duplicates(self, count: int) -> torch.Tensor:
        seqs = torch.randint(0, 2, (count, self.sequence_length), dtype=torch.long, device=self.device)
        insert_idx = torch.randint(0, self.sequence_length - len(self.pattern_tensor) + 1, (count,), device=self.device)
        for i in range(count):
            seqs[i, insert_idx[i] : insert_idx[i] + len(self.pattern_tensor)] = self.pattern_tensor
        return seqs

    def _generate_without_pattern_duplicates(self, count: int) -> torch.Tensor:
        seqs = torch.empty((count, self.sequence_length), dtype=torch.long, device=self.device)
        for i in range(count):
            while True:
                seq = torch.randint(0, 2, (self.sequence_length,), dtype=torch.long, device=self.device)
                if not self._contains_pattern(seq):
                    seqs[i] = seq
                    break
        return seqs

    def _generate_with_pattern_unique(self, count: int) -> Set[Tuple[int, ...]]:
        unique_samples: Set[Tuple[int, ...]] = set()
        while len(unique_samples) < count:
            batch_size = int((count - len(unique_samples)) * 1.5) + 10
            seqs = self._generate_with_pattern_duplicates(batch_size)
            unique_samples.update(tuple(s.tolist()) for s in seqs)
        return set(random.sample(list(unique_samples), count))

    def _generate_without_pattern_unique(self, count: int, existing_samples: Set[Tuple[int, ...]]) -> Set[Tuple[int, ...]]:
        unique_samples: Set[Tuple[int, ...]] = set()
        while len(unique_samples) < count:
            batch_size = int((count - len(unique_samples)) * 1.5) + 10
            candidates = torch.randint(0, 2, (batch_size, self.sequence_length), dtype=torch.long, device=self.device)
            for seq in candidates:
                if not self._contains_pattern(seq):
                    seq_tuple = tuple(seq.tolist())
                    if seq_tuple not in existing_samples:
                        unique_samples.add(seq_tuple)
        return set(random.sample(list(unique_samples), count))

    def _contains_pattern(self, sequence: torch.Tensor) -> bool:
        n, m = len(sequence), len(self.pattern_tensor)
        for i in range(n - m + 1):
            if torch.equal(sequence[i:i+m], self.pattern_tensor):
                return True
        return False


class PalindromeDataGenerator(BaseTensorGenerator):
    """Generates binary sequences for a palindrome task with a 50/50 split."""
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu', allow_duplicates: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0: raise ValueError("num_samples must be even.")
        self.allow_duplicates = allow_duplicates
        logger.info(f"PalindromeDataGenerator initialized, allow_duplicates={allow_duplicates}")

    def _generate_raw_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.allow_duplicates:
            palindromes = self._generate_palindromes_duplicates(self.num_positive_samples)
            non_palindromes = self._generate_non_palindromes_duplicates(palindromes)
        else:
            palindromes = self._generate_palindromes_unique(self.num_positive_samples)
            non_palindromes = self._generate_non_palindromes_unique(self.num_negative_samples, palindromes)
        return list(palindromes), list(non_palindromes)

    def _generate_palindromes(self, first_halves: torch.Tensor) -> torch.Tensor:
        L, half_len = self.sequence_length, (self.sequence_length + 1) // 2
        second_halves_rev = torch.flip(first_halves[:, :L // 2], dims=[1])
        return torch.cat([first_halves, second_halves_rev], dim=1)

    def _generate_palindromes_duplicates(self, count: int) -> torch.Tensor:
        half_len = (self.sequence_length + 1) // 2
        first_halves = torch.randint(0, 2, size=(count, half_len), device=self.device, dtype=torch.long)
        return self._generate_palindromes(first_halves)

    def _generate_non_palindromes_duplicates(self, base_palindromes: torch.Tensor) -> torch.Tensor:
        non_palindromes = base_palindromes.clone()
        half_len = (self.sequence_length + 1) // 2
        rows = torch.arange(non_palindromes.size(0), device=self.device)
        cols_to_flip = torch.randint(0, half_len, size=(non_palindromes.size(0),), device=self.device)
        non_palindromes[rows, cols_to_flip] = 1 - non_palindromes[rows, cols_to_flip]
        return non_palindromes

    def _generate_palindromes_unique(self, count: int) -> torch.Tensor:
        half_len = (self.sequence_length + 1) // 2
        if count > 2**half_len:
            raise ValueError(f"Cannot generate {count} unique palindromes; only {2**half_len} exist.")
        
        unique_halves: Set[Tuple[int, ...]] = set()
        while len(unique_halves) < count:
            batch_size = int((count - len(unique_halves)) * 1.5) + 10
            fh = torch.randint(0, 2, size=(batch_size, half_len), device=self.device, dtype=torch.long)
            unique_halves.update(tuple(row.tolist()) for row in fh)
        
        first_halves = torch.tensor(random.sample(list(unique_halves), count), dtype=torch.long, device=self.device)
        return self._generate_palindromes(first_halves)

    def _generate_non_palindromes_unique(self, count: int, existing_palindromes: torch.Tensor) -> torch.Tensor:
        nonpal_set: Set[Tuple[int, ...]] = set()
        pal_set = {tuple(p.tolist()) for p in existing_palindromes}
        while len(nonpal_set) < count:
            batch_size = int((count - len(nonpal_set)) * 1.5) + 10
            candidates = torch.randint(0, 2, (batch_size, self.sequence_length), device=self.device, dtype=torch.long)
            for seq in candidates:
                if not torch.equal(seq, torch.flip(seq, dims=[0])):
                    seq_tuple = tuple(seq.tolist())
                    if seq_tuple not in pal_set and seq_tuple not in nonpal_set:
                        nonpal_set.add(seq_tuple)
        return torch.tensor(random.sample(list(nonpal_set), count), dtype=torch.long, device=self.device)
    

def get_data_generator(target_name: str, sequence_length: int, num_samples: int) -> BaseDataGenerator:
    """
    Factory function to select and instantiate the correct data generator.

    Args:
        target_name (str): The name of the target function (e.g., 'dyck2', 'palindrome').
        sequence_length (int): The length of the input sequences (L).
        num_samples (int): The number of samples to generate.

    Returns:
        An instance of a BaseDataGenerator subclass.
    """
    # This logic is now centralized here
    if target_name == 'dyck2':
        # Dyck-2 generator has constraints on sample size for smaller lengths
        if sequence_length == 20:
            return Dyck2DataGenerator(sequence_length, num_samples, allow_duplicates=True)
        else:
            return Dyck2DataGenerator(sequence_length, num_samples)
    
    if target_name in ['patternmatch1', 'patternmatch2']:
        if target_name == 'patternmatch2':
            return PatternBasedDataGenerator(sequence_length, num_samples, pattern_string='00111111')
        else:
            return PatternBasedDataGenerator(sequence_length, num_samples)  # defaults to '10101010'
    
    if target_name == "palindrome":
        if sequence_length == 20:
            return PalindromeDataGenerator(sequence_length, num_samples, allow_duplicates=True)
        else:
            return PalindromeDataGenerator(sequence_length, num_samples)
        
    if target_name == "prime_decimal":
        return PrimeDataGenerator(sequence_length, num_samples)
        
    if target_name == "prime_decimal_tf_check":
        return PrimeDecimalTailRestrictedDataGenerator(sequence_length, num_samples, allow_leading_zeros=False)
    
    # Default to BinaryDataGenerator for all other function names
    if target_name in TARGET_FUNCTIONS:
        return BinaryDataGenerator(target_name, sequence_length, num_samples)

    raise ValueError(f"No data generator found for target function '{target_name}'")

def create_stratified_splits(
    all_samples: List[Dict[str, Any]],
    train_size: int,
    val_size: int,
    test_size: int,
    device: str = 'cpu'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Creates stratified train, validation, and test splits from a generated dataset.

    This utility ensures that each split maintains a balanced 50/50 class distribution,
    which is critical for the experiments.

    Args:
        all_samples: A list of generated data samples, where each sample is a
                     dictionary like {'Input': np.array, 'Output': '0' or '1'}.
        train_size: The desired number of samples in the training set.
        val_size: The desired number of samples in the validation set.
        test_size: The desired number of samples in the test set.
        device: The torch device to use for tensor operations.

    Returns:
        A tuple containing three lists of dictionaries:
        (train_split, validation_split, test_split).
    """
    if train_size % 2 != 0:
        raise ValueError("train_size must be even for a balanced split.")
    if val_size % 2 != 0:
        raise ValueError("val_size must be even for a balanced split.")

    # Convert the list of dicts to a format that's easier to split
    original_indices = list(range(len(all_samples)))
    all_labels = torch.tensor([int(s['Output']) for s in all_samples], device=device)

    # Separate indices by class
    indices_0 = torch.where(all_labels == 0)[0]
    indices_1 = torch.where(all_labels == 1)[0]

    # Deterministic shuffle of indices for each class
    shuffled_indices_0 = indices_0[torch.randperm(len(indices_0), device=device)]
    shuffled_indices_1 = indices_1[torch.randperm(len(indices_1), device=device)]

    # Calculate samples per class for each split
    train_per_class = train_size // 2
    val_per_class = val_size // 2

    if len(shuffled_indices_0) < train_per_class + val_per_class:
        raise ValueError("Not enough samples of class 0 for the requested train/val split size.")
    if len(shuffled_indices_1) < train_per_class + val_per_class:
        raise ValueError("Not enough samples of class 1 for the requested train/val split size.")

    # Create train indices
    train_indices_0 = shuffled_indices_0[:train_per_class]
    train_indices_1 = shuffled_indices_1[:train_per_class]
    train_indices = torch.cat([train_indices_0, train_indices_1])
    # Final shuffle to mix classes within the training set
    train_indices = train_indices[torch.randperm(len(train_indices), device=device)]

    # Create validation indices
    val_indices_0 = shuffled_indices_0[train_per_class : train_per_class + val_per_class]
    val_indices_1 = shuffled_indices_1[train_per_class : train_per_class + val_per_class]
    val_indices = torch.cat([val_indices_0, val_indices_1])

    # Create test indices from the remainder
    test_indices_0 = shuffled_indices_0[train_per_class + val_per_class:]
    test_indices_1 = shuffled_indices_1[train_per_class + val_per_class:]
    test_indices = torch.cat([test_indices_0, test_indices_1])

    # Reconstruct the splits using the original list and the selected indices
    train_split = [all_samples[i] for i in train_indices.tolist()]
    val_split = [all_samples[i] for i in val_indices.tolist()]
    test_split = [all_samples[i] for i in test_indices.tolist()]

    # Sanity checks
    assert len(train_split) == train_size
    assert len(val_split) == val_size
    assert len(test_split) == test_size

    return train_split, val_split, test_split