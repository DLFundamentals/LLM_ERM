import torch
import operator as op

def full(x, const, device='cpu'):
    return torch.full_like(x, const, dtype=torch.float, device=device)

def indices(x, device='cpu'):
    return torch.arange(len(x), dtype=torch.float, device=device)

def tok_map(x, func, device='cpu'):
    return torch.tensor([func(xi) for xi in x], dtype=torch.float, device=device)

def seq_map(x, y, func, device='cpu'):
    return torch.tensor([func(xi, yi) for xi, yi in zip(x, y)], dtype=torch.float, device=device)

def select(k, q, pred, causal=True, device='cpu'):
    s = len(k)
    A = torch.zeros((s, s), dtype=torch.bool, device=device)
    for qi in range(s):
        for kj in range(qi+1 if causal else s):
            A[qi, kj] = pred(k[kj].item(), q[qi].item())
    return A

def sel_width(A, device='cpu'):
    return torch.matmul(A, torch.ones(len(A), dtype=torch.float, device=device))

def aggr_mean(A, v, default=0, device='cpu'):
    out = torch.matmul(A, v)
    norm = sel_width(A, device=device)
    out = torch.div(out, norm, out=torch.full_like(v, default, dtype=torch.float, device=device), where=(norm != 0))
    return out.to(torch.float)

def aggr_max(A, v, default=0, device='cpu'):
    out = torch.full_like(v, default, device=device)
    for i, row in enumerate(A):
        idxs = torch.nonzero(row).squeeze()
        if len(idxs) > 0:
            out[i] = torch.max(v[idxs])
    return out.to(torch.float)

def aggr_min(A, v, default=0, device='cpu'):
    return -aggr_max(A, -v, -default, device=device)

def aggr(A, v, default=0, reduction='mean', device='cpu'):
    if reduction == 'mean':
        return aggr_mean(A, v, default, device=device)
    elif reduction == 'max':
        return aggr_max(A, v, default, device=device)
    elif reduction == 'min':
        return aggr_min(A, v, default, device=device)

def kqv(k, q, v, pred, default=0, reduction='mean', device='cpu'):
    return aggr(select(k, q, pred, device=device), v, default=default, reduction=reduction, device=device)

# Define comparison operators
equals, leq, lt, geq, gt = op.eq, op.le, op.lt, op.ge, op.gt

def shift_right(x, n, default=0, device='cpu'):
    # shifts sequence x to the right by n positions
    return kqv(indices(x, device=device) + n, indices(x, device=device), x, equals, default=default, device=device)

def cumsum(bool_array, device='cpu'):
    # returns number of previous True elements in bool_array
    return sel_width(select(bool_array, bool_array, lambda k, q: k, device=device), device=device)

def where(condition, x_if, y_else, device='cpu'):
    # equivalent to np.where(condition, x_if, y_else)
    x_masked = seq_map(x_if, condition, lambda x, m: x if m else 0, device=device)
    y_masked = seq_map(y_else, condition, lambda y, m: y if not m else 0, device=device)
    return seq_map(x_masked, y_masked, lambda x, y: x if y == 0 else y, device=device)

def mask(x, bool_mask, mask_val=0, device='cpu'):
    # equivalent to x*bool_mask + default*(~bool_mask)
    return where(bool_mask, x, full(x, mask_val, device=device), device=device)

def maximum(x, device='cpu'):
    return kqv(x, x, x, lambda k, q: True, reduction='max', device=device)

def minimum(x, device='cpu'):
    return -maximum(-x, device=device)

def argmax(x, device='cpu'):
    mm = maximum(x, device=device)
    return kqv(mm, x, indices(x, device=device), reduction='max', device=device)

def argmin(x, device='cpu'):
    return argmax(-x, device=device)

def num_prev(x, queries, device='cpu'):
    # output[i] = number of previous elements of x equal to queries[i], inclusive
    return sel_width(select(x, queries, equals, device=device), device=device)

def has_seen(x, queries, device='cpu'):
    return kqv(x, queries, full(x, 1, device=device), equals, default=0, device=device)

def firsts(x, queries, default=-1, device='cpu'):
    # find the index of the first occurrence of each query[i] in x
    # out[i] := np.flatnonzero(x[:i+1] == queries[i]).min()
    return kqv(x, queries, indices(x, device=device), equals, default=default, reduction='min', device=device)

def lasts(x, queries, default=-1, device='cpu'):
    # find the index of the last occurrence of each query[i] in x
    # out[i] := np.flatnonzero(x[:i+1] == queries[i]).max()
    return kqv(x, queries, indices(x, device=device), equals, default=default, reduction='max', device=device)

def index_select(x, idx, default=0, device='cpu'):
    # indexes into sequence x, via index sequence idx
    # i.e., return x[idx] if idx[i] <= i else default
    return kqv(indices(x, device=device), idx, x, equals, default=default, device=device)

def first_true(x, default=-1, device='cpu'):
    # returns the index of the first true value in x
    seen_true = kqv(x, full(x, 1, device=device), full(x, 1, device=device), equals, default=0, device=device)
    first_occ = seq_map(seen_true, shift_right(seen_true, 1, device=device), lambda curr, prev: curr and not prev, device=device)
    return kqv(first_occ, full(x, 1, device=device), indices(x, device=device), equals, default=default, device=device)

def induct_kqv(k, q, v, offset, default=0, null_val=-999, device='cpu'):
    # get value of v at index of: first occurrence of q[i] found in k (if found) + offset.
    # (excludes the last OFFSET tokens of k from matching)
    # null_val is a special token that cannot appear in k or q; used to prevent accidental matches
    indices_to_copy = firsts(shift_right(k, offset, default=null_val, device=device), q, default=null_val, device=device)
    copied_values = index_select(v, indices_to_copy, default=default, device=device)
    return copied_values

def induct(k, q, offset, default=0, null_val=-999, device='cpu'):
    return induct_kqv(k, q, k, offset=offset, default=default, null_val=null_val, device=device)

def induct_prev(k, q, offset, default=0, null_val=-999, device='cpu'):
    # A version of induct for negative offsets.
    indices_to_copy = firsts(k, q, default=null_val, device=device) + offset
    copied_values = index_select(k, indices_to_copy, default=default, device=device)
    return copied_values