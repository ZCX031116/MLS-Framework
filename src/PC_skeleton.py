import numpy as np
from itertools import combinations
from math import atanh, sqrt
from typing import Dict, Tuple, List, Optional

# ---------- Fisher-Z CI test for Gaussian ----------
def _partial_corr_from_cov(cov: np.ndarray, i: int, j: int, S: List[int]) -> float:
    if len(S) == 0:
        denom = np.sqrt(cov[i, i] * cov[j, j])
        return 0.0 if denom == 0 else cov[i, j] / denom

    idx = [i, j] + list(S)
    sub = cov[np.ix_(idx, idx)]
    try:
        prec = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        eps = 1e-8
        prec = np.linalg.inv(sub + eps * np.eye(sub.shape[0]))

    p_ij = prec[0, 1]
    p_ii = prec[0, 0]
    p_jj = prec[1, 1]
    denom = np.sqrt(max(p_ii * p_jj, 0.0))
    if denom == 0:
        return 0.0
    return float(-p_ij / denom)

def fisher_z_p_value(rho: float, n: int, cond_set_size: int) -> float:
    rho = float(np.clip(rho, -0.999999, 0.999999))
    z = atanh(rho) * sqrt(max(n - cond_set_size - 3, 1))
    from math import erf
    def phi(x):  # standard normal CDF
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    p = 2.0 * (1.0 - phi(abs(z)))
    return float(np.clip(p, 0.0, 1.0))

# ---------- PC skeleton ----------
def pc_skeleton(
    data: np.ndarray,
    alpha: float = 0.01,
    max_cond_set: int = 6,
    *,
    stable: bool = True,
    candidate_cap: Optional[int] = 12,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, ...]]]:
    X = np.asarray(data, dtype=float)
    n, d = X.shape

    # standardize
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
    cov = np.cov(X, rowvar=False, bias=False)

    # abs corr for neighbor ranking (optional cap)
    denom = np.sqrt(np.outer(np.diag(cov), np.diag(cov))) + 1e-12
    corr = cov / denom
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)

    H = np.ones((d, d), dtype=int)
    np.fill_diagonal(H, 0)

    sep_sets: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    def neighbors(u: int) -> List[int]:
        nbr = np.where(H[u] == 1)[0].tolist()
        if candidate_cap is not None and len(nbr) > candidate_cap:
            nbr = sorted(nbr, key=lambda v: abs_corr[u, v], reverse=True)[:candidate_cap]
        return nbr

    l = 0
    while True:
        if verbose:
            print(f"[PC] l={l}, edges={int(H.sum()//2)}")
        deletions = []
        any_tested = False

        edges = [(i, j) for i in range(d) for j in range(i+1, d) if H[i, j] == 1]
        for i, j in edges:
            adj_i = neighbors(i)
            if j in adj_i:
                adj_i.remove(j)
            if len(adj_i) < l:
                continue

            any_tested = True
            found_sep = None
            for S in combinations(adj_i, l):
                rho = _partial_corr_from_cov(cov, i, j, list(S))
                pval = fisher_z_p_value(rho, n=n, cond_set_size=l)
                if pval > alpha:
                    found_sep = tuple(S)
                    break

            if found_sep is not None:
                if stable:
                    deletions.append((i, j))
                else:
                    H[i, j] = H[j, i] = 0
                sep_sets[(i, j)] = found_sep
                if verbose:
                    print(f"  remove {i}-{j} sep={found_sep}")

        if stable and deletions:
            for i, j in deletions:
                H[i, j] = H[j, i] = 0

        l += 1
        if l > max_cond_set or not any_tested:
            break

    return H, sep_sets

# ---------- Paper-aligned H construction + h_j^+ expansion ----------
def build_H_pc_plus(
    data: np.ndarray,
    alpha: float = 0.01,
    max_cond_set: int = 6,
    candidate_cap: int = 12,
    *,
    extend_one: bool = True,
    extra_parent_cap: int = 8,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Returns:
      H: symmetric skeleton (d,d)
      extra_parents: dict j -> list of extra candidates outside H[:,j] (for h_j^+).
    """
    H, _ = pc_skeleton(data, alpha=alpha, max_cond_set=max_cond_set,
                       stable=True, candidate_cap=candidate_cap, verbose=verbose)
    H = np.maximum(H, H.T)
    np.fill_diagonal(H, 0)

    X = np.asarray(data, dtype=float)
    # correlation ranking for extra parents (cheap & paper-compatible as heuristic for h_j^+)
    Xs = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
    C = np.abs(np.corrcoef(Xs, rowvar=False))
    np.fill_diagonal(C, 0.0)

    d = H.shape[0]
    extra_parents: Dict[int, List[int]] = {}

    if not extend_one:
        for j in range(d):
            extra_parents[j] = []
        return H, extra_parents

    for j in range(d):
        hj = set(np.where(H[:, j] == 1)[0].tolist())
        outside = [i for i in range(d) if i != j and i not in hj]
        # pick top outside by abs corr to j
        outside_sorted = sorted(outside, key=lambda i: C[i, j], reverse=True)[:extra_parent_cap]
        extra_parents[j] = outside_sorted

    return H, extra_parents
