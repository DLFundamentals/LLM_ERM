# /src/target_functions.py
"""
Defines a canonical set of target functions for generating ground truth data.
Each function takes a PyTorch tensor and a device string, returning a tensor of 0s and 1s.
"""
import torch
import torch.nn.functional as F
import hashlib
from typing import Callable, Dict, Any
from sympy import isprime

# --- Core Implementations ---

def parity_all(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of all coordinates."""
    return (v.sum(dim=1) % 2).long()

def parity_first_half(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of the first half of the coordinates."""
    return (v[:, :v.shape[1]//2].sum(dim=1) % 2).long()

def automata_parity(v: torch.Tensor, device: str) -> torch.Tensor:
    """Applies a rule-based transformation on a sliding window of size 3 (Rule 30-like)."""
    inds = F.pad(v.float(), (1, 1), 'constant', 0).unfold(1, 3, 1).matmul(torch.tensor([4, 2, 1], device=device, dtype=torch.float))
    rule = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], device=device, dtype=torch.float)
    return (rule[inds.long()].sum(1) % 2).long()

def sha256_parity(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of '1' bits in the SHA-256 hash of the binary string."""
    results = []
    for row in v:
        row_str = ''.join(map(str, row.int().tolist()))
        hashed = hashlib.sha256(row_str.encode()).hexdigest()
        binary_hash = bin(int(hashed, 16))[2:]
        results.append(binary_hash.count('1') % 2)
    return torch.tensor(results, device=device, dtype=torch.long)

def palindrome(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if each binary sequence in the batch is a palindrome."""
    flipped_v = torch.flip(v, dims=[1])
    return torch.all(v == flipped_v, dim=1).long()

def dyck2(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks for valid Dyck-2 sequences, e.g., '()[]'."""
    pmap = {"00": "(", "01": ")", "10": "[", "11": "]"}
    match = {')': '(', ']': '['}
    def to_paren(row):
        return "".join(pmap.get(f"{row[i]}{row[i+1]}", "?") for i in range(0, len(row), 2))

    def is_valid(s):
        stack = []
        for c in s:
            if c in match.values(): stack.append(c)
            elif not stack or stack.pop() != match[c]: return 0
        return int(not stack)

    return torch.tensor([is_valid(to_paren(r.tolist())) for r in v], device=device, dtype=torch.long)

def parity_rand_3(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of 3 random but fixed coordinates."""
    torch.manual_seed(42)
    idx = torch.randperm(v.shape[1])[:3]
    return (v[:, idx].sum(dim=1) % 2).long()

def parity_rand_10(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of 10 random but fixed coordinates."""
    torch.manual_seed(42)
    idx = torch.randperm(v.shape[1])[:10]
    return (v[:, idx].sum(dim=1) % 2).long()

def patternmatch1(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks for the presence of the pattern '10101010'."""
    pattern_str = '10101010'
    return torch.tensor([pattern_str in "".join(map(str, row.tolist())) for row in v], dtype=torch.long, device=device)

def patternmatch2(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks for the presence of the pattern '00111111'."""
    pattern_str = '00111111'
    return torch.tensor([pattern_str in "".join(map(str, row.tolist())) for row in v], dtype=torch.long, device=device)

def prime_decimal(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if a decimal number is prime."""
    numbers = [int("".join(map(str, row.tolist()))) for row in v]
    return torch.tensor([isprime(n) for n in numbers], dtype=torch.long, device=device)

def prime_decimal_tf_check(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if a decimal number is prime (identical to prime_decimal, used for different generator)."""
    return prime_decimal(v, device)


# --- Canonical Mapping ---
TARGET_FUNCTIONS: Dict[str, Callable[[torch.Tensor, str], torch.Tensor]] = {
    'parity_all': parity_all,
    'parity_first_half': parity_first_half,
    'patternmatch1': patternmatch1,          # '10101010'
    'patternmatch2': patternmatch2,          # '00111111'
    'parity_rand_3': parity_rand_3,
    'parity_rand_10': parity_rand_10,
    'palindrome': palindrome,
    'dyck2': dyck2,
    'prime_decimal': prime_decimal,
    'automata_parity': automata_parity,
    'prime_decimal_tf_check': prime_decimal_tf_check,
    'sha256_parity': sha256_parity,
}

EXPERIMENT_FUNCTION_MAPPING: Dict[str, str] = {
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

EXPERIMENT_FUNCTION_METADATA: Dict[str, Dict[str, Any]] = {
    "fn_h": {
        "lengths": [100, 80, 60, 40, 20]
    }
}