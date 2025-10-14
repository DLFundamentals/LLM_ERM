# src/target_functions.py
"""
Defines potential target functions used for generating the ground truth data.
Each function takes a PyTorch tensor of binary inputs and the device string,
and returns a tensor of long integers (0 or 1).
"""
import torch
import torch.nn.functional as F
import hashlib
from typing import Callable, Dict

def parity_first_half(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of the first half of the coordinates."""
    return (v[:, :v.shape[1]//2].sum(dim=1) % 2).long()

def automata_parity(v, device):
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

def is_palindrome(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if each binary sequence in the batch is a palindrome."""
    flipped_v = torch.flip(v, dims=[1])
    comparison = (v == flipped_v)
    palindrome_check = torch.all(comparison, dim=1)
    return palindrome_check.long()

def parity_all(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of all coordinates."""
    return (v.sum(dim=1) % 2).long()

def dyck2(v: torch.Tensor, device: str) -> torch.Tensor:
    """
    Returns (N,) tensor where each entry is 1 if the decoded paren sequence is valid, else 0.
    """
    pmap = {"00": "(", "01": ")", "10": "[", "11": "]"}
    match = {')': '(', ']': '['}
    to_paren = lambda row: "".join(pmap[f"{row[i]}{row[i+1]}"] for i in range(0, len(row), 2))
    
    def is_valid(s):
        stack = []
        for c in s:
            if c in match.values(): stack.append(c)
            elif not stack or stack.pop() != match[c]: return 0
        return int(not stack)
    
    return torch.tensor([is_valid(to_paren(r)) for r in v.tolist()], device=device, dtype=torch.float)

def parity_rand_10(v, device):
    torch.manual_seed(42)
    idx = torch.randperm(v.shape[1])[:10]
    return (v[:, idx].sum(dim=1) % 2).long()

def parity_rand_3(v, device):
    torch.manual_seed(42)
    idx = torch.randperm(v.shape[1])[:3]
    return (v[:, idx].sum(dim=1) % 2).long()

def patternmatch1(v: torch.Tensor, device: str) -> torch.Tensor:
    pattern = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=v.dtype, device=device)
    N, L = v.shape
    match_length = pattern.size(0)
    results = torch.zeros(N, dtype=torch.bool, device=device)

    for i in range(L - match_length + 1):
        window = v[:, i:i + match_length]  # (N, match_length)
        match = (window == pattern).all(dim=1)  # (N,)
        results = results | match  # now valid because both are bool

    return results.float()

def patternmatch2(v: torch.Tensor, device: str) -> torch.Tensor:
    pattern = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1], dtype=v.dtype, device=device)
    N, L = v.shape
    match_length = pattern.size(0)
    results = torch.zeros(N, dtype=torch.bool, device=device)

    for i in range(L - match_length + 1):
        window = v[:, i:i + match_length]  # (N, match_length)
        match = (window == pattern).all(dim=1)  # (N,)
        results = results | match  # now valid because both are bool

    return results.float()

def parity_rand_3_fixed(v, device):
    torch.manual_seed(42)
    idx = torch.randperm(10)[:3]
    return (v[:, idx].sum(dim=1) % 2).long()


TARGET_FUNCTIONS: Dict[str, Callable[[torch.Tensor, str], torch.Tensor]] = {
    'parity_first_half': parity_first_half,
    'automata_parity': automata_parity,
    'sha256_parity': sha256_parity,
    'is_palindrome': is_palindrome,
    'parity_all': parity_all,
    'dyck2': dyck2,
    'parity_rand_10' : parity_rand_10,
    'parity_rand_3' : parity_rand_3,
    'patternmatch1' : patternmatch1,
    'patternmatch2' : patternmatch2,
    'parity_rand_3_fixed' : parity_rand_3_fixed,
}