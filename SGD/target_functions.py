import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasp_functions import *
import hashlib

# the parity of the first half of the coordinates
def func1(v, device):
    return v[:, :v.shape[1]//2].sum(1) % 2

# calculates the parity of the sum of the maximum values in each sliding window of the first half
def func2(v, device):
    return (v.unfold(1, v.shape[1]//2, 1).sum(2).max(1).values % 2)

# applies a rule-based transformation on a sliding window of size 3
def func3(v, device):
    inds = F.pad(v.float(), (1, 1), 'constant', 0).unfold(1, 3, 1).matmul(torch.tensor([4, 2, 1], device=device, dtype=torch.float))
    rule = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], device=device, dtype=torch.float)
    return (rule[inds.long()].sum(1) % 2).long()

# computes the parity (even or odd) of the number of '1' bits in the SHA-256 hash
def func4(v, device):
    def f(row):
        row_st = ''.join(map(str, row.int().tolist()))
        row_bin = bin(int(hashlib.sha256(row_st.encode()).hexdigest(), 16))[2:]
        return row_bin.count('1') % 2

    return torch.tensor([f(row) for row in v], device=device)

# partiy of the even ones
def func6(v, device):
    return v[:, ::2].sum(1) % 2


def func7(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of all coordinates."""
    # Parity is typically well-balanced (50/50) for random inputs.
    return (v.sum(dim=1) % 2).long()

def func8(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of the coordinates at even indices (0, 2, 4...)."""
    # Parity of a random subset is typically well-balanced.
    return (v[:, ::2].sum(dim=1) % 2).long()

def func9(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of the coordinates at odd indices (1, 3, 5...)."""
    # Parity of a random subset is typically well-balanced.
    if v.shape[1] < 2: # Handle vectors shorter than 2
        return torch.zeros(v.shape[0], device=device, dtype=torch.long)
    return (v[:, 1::2].sum(dim=1) % 2).long()

def func10(v: torch.Tensor, device: str) -> torch.Tensor:
    """Returns the value of the first bit."""
    # If input bits are 50/50 random, output will be 50/50. Extremely simple.
    return v[:, 0].long()

def func11(v: torch.Tensor, device: str) -> torch.Tensor:
    """Returns the value of the last bit."""
    # If input bits are 50/50 random, output will be 50/50. Extremely simple.
    return v[:, -1].long()

def func12(v: torch.Tensor, device: str) -> torch.Tensor:
    """Returns the value of the middle bit (floor index for even length)."""
    # If input bits are 50/50 random, output will be 50/50. Extremely simple.
    middle_index = v.shape[1] // 2
    return v[:, middle_index].long()

def func13(v: torch.Tensor, device: str) -> torch.Tensor:
    """Returns the XOR (parity) of the first and last bits."""
    # If first/last bits are independent random 50/50, XOR is 50/50. Simple.
    # (a != b) is equivalent to (a + b) % 2 for binary inputs
    return (v[:, 0] != v[:, -1]).long()

def func14(v: torch.Tensor, device: str) -> torch.Tensor:
    """Returns 1 if the first and last bits are equal, 0 otherwise."""
    # Opposite of xor_first_last, also likely 50/50. Simple.
    return (v[:, 0] == v[:, -1]).long()

def func15(v, device):
    """ random parity 3 """
    torch.manual_seed(42)
    idx = torch.randperm(v.shape[1])[:3]
    return (v[:, idx].sum(dim=1) % 2).long()

def func16(v: torch.Tensor, device: str) -> torch.Tensor:
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

def func17(v, device):
    reversed_v = torch.flip(v, dims=[1])
    is_pal = torch.all(v == reversed_v, dim=1)
    return is_pal.long()

def func18(v, device):
    torch.manual_seed(42)
    idx = torch.randperm(v.shape[1])[:10]
    return (v[:, idx].sum(dim=1) % 2).long()

from sympy import isprime

def func19(v: torch.Tensor, device="cpu") -> torch.Tensor:
    numbers = [int("".join(map(str, row.tolist()))) for row in v]
    return torch.tensor([isprime(n) for n in numbers], dtype=torch.long, device=device)

def func20(v: torch.Tensor, device="cpu") -> torch.Tensor:
    pattern_str = '10101010'
    return torch.tensor([pattern_str in "".join(map(str, row.tolist())) for row in v], dtype=torch.long, device=device)

def func21(v: torch.Tensor, device="cpu") -> torch.Tensor:
    pattern_str = '00111111'
    return torch.tensor([pattern_str in "".join(map(str, row.tolist())) for row in v], dtype=torch.long, device=device)

def func22(v, device):
    from sympy import isprime
    numbers = [int("".join(map(str, row.tolist()))) for row in v]
    return torch.tensor([isprime(n) for n in numbers], dtype=torch.long, device=device)