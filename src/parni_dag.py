# -*- coding: utf-8 -*-
"""
Minimal PARNI-DAG implementation (paper-faithful core only)

This file is a *reduced* version derived from your parni_dag_2.py, keeping only:
  - BGe score backend (same as long_multilevel_splitting_frame usage)
  - Core PARNI-DAG neighborhood proposal (sample k, update_LA_DAG, MH ratio pieces)
  - Optional Eq.(9) PIPs adaptation (Appendix B phi schedule)
  - Optional Eq.(13) omega (neighborhood thinning) Robbins–Monro update
  - The small interface required by long_multilevel_splitting_frame:
      parni_prepare_context, parni_make_LA_from_G, parni_step_one, parni_update_pips_eq9

Everything else (warm-start parent-set enumeration from skeleton H, g-prior backend,
extra utilities/experiments) is intentionally removed to simplify debugging.

Data orientation matches the R implementation:
  X_p_n.shape == (p, n)  (rows = variables, cols = samples)
"""
from __future__ import annotations

import numpy as np
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, List
from collections import deque
try:
    from src.bge import BGe  # type: ignore
except Exception:  # pragma: no cover
    BGe = object  # type: ignore


# =============================================================================
# 1) BGe adapter + log marginal likelihood cache (node-wise decomposable)
# =============================================================================
class BGEAdapter:
    """
    Adapter for trust.utils.bge.BGe.

    Assumptions:
      - bge_obj.mll(G, X) computes the whole-graph log marginal likelihood.
      - bge_obj._mll_per_variable(i, parents_vec, R, N) computes local mll for node i.
      - Your data for BGe is (N, d). Internally we store (p, n) and transpose.
    """
    def __init__(self, bge_obj: BGe, X_p_n: np.ndarray):
        assert X_p_n.ndim == 2
        self.p = int(X_p_n.shape[0])
        self.X_n_d = X_p_n.T
        self.N = int(self.X_n_d.shape[0])
        self.bge = bge_obj
        self.R = self.bge.calc_R(self.X_n_d)

    def local_llh(self, j: int, G: np.ndarray) -> float:
        parents_vec = G[:, j].astype(int)
        return float(self.bge._mll_per_variable(j, parents_vec, self.R, self.N))

@dataclass
class HyperPar:
    X: np.ndarray                       # (p, n)
    h: float | Tuple[float, float]      # edge prior (Bernoulli(h) or Beta-Binomial(a,b) on |E|)
    p: int
    n: int
    max_p: int
    log_llh: Callable                   # function(LA, hyper_par) -> LA
    log_llh_update: Callable            # function(changes, LA_old, LA, hyper_par) -> LA
    log_m_prior: Callable               # function(p_gam, h, max_p) -> float
    tables: BGEAdapter                  # BGeAdapter (stores R, N)
    XtX: Optional[np.ndarray] = None    # (p, p) cross-product for warm-start heuristics

@dataclass
class LAState:
    curr: np.ndarray
    p_gam: int
    llh: float = 0.0
    lmp: float = 0.0
    log_post: float = 0.0
    A: Optional[np.ndarray] = None      # per-node contributions

def log_llh_BGE(LA: LAState, hyper_par: HyperPar) -> LAState:
    adapter: BGEAdapter = hyper_par.tables
    G = LA.curr
    p = hyper_par.p
    A = np.zeros(p, dtype=float)
    for j in range(p):
        A[j] = adapter.local_llh(j, G)
    LA.A = A
    LA.llh = float(np.sum(A))
    LA.p_gam = int(G.sum())
    return LA

def log_llh_BGE_update_table(changes: np.ndarray, LA_old: LAState, LA: LAState, hyper_par: HyperPar) -> LAState:
    adapter: BGEAdapter = hyper_par.tables
    G = LA.curr
    p = hyper_par.p

    # copy old cache; if missing, recompute from scratch
    if LA_old.A is None:
        A = np.zeros(p, dtype=float)
        for j in range(p):
            A[j] = adapter.local_llh(j, G)
        LA.A = A
        LA.llh = float(np.sum(A))
        LA.p_gam = int(G.sum())
        return LA

    A = LA_old.A.copy()
    cols = np.unique(np.array(changes, dtype=int))

    # tolerate R-style 1-based
    cols = np.array([c - 1 if (1 <= c <= p) else c for c in cols], dtype=int)
    cols = cols[(cols >= 0) & (cols < p)]
    for j in cols:
        A[j] = adapter.local_llh(int(j), G)

    LA.A = A
    LA.llh = float(np.sum(A))
    LA.p_gam = int(G.sum())
    return LA

# =============================================================================
# 2) DAG utilities
# =============================================================================
def has_path(G: np.ndarray, start_node: int, end_node: int, exclude_edge: Optional[Tuple[int, int]] = None) -> bool:
    """
    使用 BFS 快速检查是否存在从 start_node 到 end_node 的路径。
    用于替代全图 Kahn 算法。
    """
    if start_node == end_node:
        return True
        
    d = G.shape[0]
    visited = np.zeros(d, dtype=bool)
    queue = deque([start_node])
    visited[start_node] = True
    
    ex_u, ex_v = (-1, -1) if exclude_edge is None else exclude_edge

    while queue:
        curr = queue.popleft()
        # 找出所有子节点
        children = np.where(G[curr, :] == 1)[0]
        
        for child in children:
            # 如果这条边是被“排除”的（用于翻转检测），则跳过
            if curr == ex_u and child == ex_v:
                continue
            
            if child == end_node:
                return True
            
            if not visited[child]:
                visited[child] = True
                queue.append(child)
    return False

def is_valid_move_incremental(G: np.ndarray, i: int, j: int, move_type: str) -> bool:
    """
    基于当前图 G，判断对边 (i, j) 进行操作是否合法。
    move_type: 'add' (加边 i->j), 'del' (删边), 'rev' (翻转 i->j 为 j->i)
    """
    if move_type == 'del':
        return True
    
    if move_type == 'add':
        # 如果要加 i->j，必须保证没有 j->...->i 的路径
        return not has_path(G, start_node=j, end_node=i)
        
    if move_type == 'rev':
        # 如果要翻转 i->j 为 j->i，相当于：先删 i->j，再加 j->i。
        # 需要检查是否存在 i->...->j 的路径（不走直连边）
        return not has_path(G, start_node=i, end_node=j, exclude_edge=(i, j))
        
    return False
    
def sample_ind_DAG(
    whe_sam: bool,
    probs: np.ndarray,
    samples: Optional[np.ndarray] = None,
    log: bool = False,
    rng: Optional[np.random.Generator] = None,
):
    """
    Sample k ~ Bernoulli(probs) entrywise over directed edges using *F-order* flattening
    to match the original R encoding: idx = i + d*j (i->j).

    Returns:
      {"prob": log p(k) if log=True else p(k), "sample": samples_idx}
    """
    if rng is None:
        rng = np.random.default_rng()

    d = probs.shape[0]
    if whe_sam:
        draws = (rng.random((d, d)) < probs).astype(int)
        samples = np.where(draws.ravel(order="F") == 1)[0]

    if log:
        if samples is None or samples.size == 0:
            prob = 0.0
        else:
            flat = probs.ravel(order="F")
            prob = float(np.sum(np.log(flat[samples] + 1e-300)))
    else:
        if samples is None or samples.size == 0:
            prob = 1.0
        else:
            flat = probs.ravel(order="F")
            prob = float(np.prod(flat[samples]))
    return {"prob": prob, "sample": samples}

# =============================================================================
# 3) Priors and LA construction
# =============================================================================
def _log_m_prior_beta_binom(p_gam: int, hval, mp: int) -> float:
    """
    Prior over number of edges |E| (up to additive constant):
      - Bernoulli(h) iid edges: |E| log h + (mp-|E|) log(1-h)
      - Beta-Binomial(a,b) on |E|: log Beta(a+|E|, b+mp-|E|) + const
    """
    if isinstance(hval, (tuple, list)):
        a, b = hval
        return float(
            np.log(np.math.gamma(p_gam + a))
            + np.log(np.math.gamma(mp - p_gam + b))
            - np.log(np.math.gamma(mp + a + b))
        )
    h = float(hval)
    h = min(max(h, 1e-12), 1 - 1e-12)
    return float(p_gam * np.log(h) + (mp - p_gam) * np.log(1 - h))

def compute_LA_DAG(gamma: np.ndarray, hyper_par: HyperPar) -> LAState:
    LA = LAState(curr=gamma.copy().astype(int), p_gam=int(gamma.sum()))
    LA = hyper_par.log_llh(LA, hyper_par)
    log_m_prior = hyper_par.log_m_prior(LA.p_gam, hyper_par.h, hyper_par.max_p)
    LA.lmp = log_m_prior
    LA.log_post = LA.llh + log_m_prior
    return LA

# =============================================================================
# 4) Core PARNI neighborhood update (update_LA_DAG)
# =============================================================================
def _get_moves() -> np.ndarray:
    # (0,0), (1,0), (0,1), (1,1)
    return np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=int)

def _logsumexp(logw: np.ndarray) -> float:
    m = np.max(logw)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(logw - m))))

def _safe_log_odds(p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p) - np.log(1 - p))

def update_LA_DAG(
    LA: LAState,
    k: np.ndarray,
    hyper_par: HyperPar,
    bal_fun: Callable[[float], float],
    PIPs: np.ndarray,
    thinning_rate: float,
    rng: Optional[np.random.Generator] = None,
):
    """
    One informed sub-proposal sequence over the neighborhood indices in k.

    This is a log-domain implementation of the R code:
      - groups (i->j, j->i) pairs if both present in k
      - for each group, enumerate local candidate graphs (2 or 4 states)
      - weight by Hastings balancing function g(x)=min(1,x) (log_g = min(0, L))
      - sample a local move and accumulate log q forward/reverse and normalization constants

    Returns dict with:
      LA_prop, prob_prop, rev_prob_prop, prod_bal_con, rev_prod_bal_con, thinned_k_size, ...
    """
    if rng is None:
        rng = np.random.default_rng()

    LA_temp = LA
    log_post_temp = float(LA.log_post)

    max_p, p, h = hyper_par.max_p, hyper_par.p, hyper_par.h
    log_llh_update = hyper_par.log_llh_update
    log_m_prior = hyper_par.log_m_prior

    d = p
    if k is None or len(k) == 0:
        return dict(
            LA_prop=LA,
            JD=0,
            acc_rate=0.0,
            thinned_k_size=0,
            total_k_size=0,
            n_eval=0,
            prob_prop=0.0,
            rev_prob_prop=0.0,
            prod_bal_con=0.0,
            rev_prod_bal_con=0.0,
        )

    # k contains F-order linear indices: idx = i + d*j
    k_idx = np.array(k, dtype=int).ravel()
    selected = set(int(x) for x in k_idx)

    # group (i->j) with (j->i) if both present; otherwise singleton
    ij_pairs = []
    processed: set[int] = set()
    for idx0 in sorted(selected):
        if idx0 in processed:
            continue
        i = int(idx0 % d)
        j = int(idx0 // d)
        if i == j:
            processed.add(idx0)
            continue
        idx_rev = int(j + d * i)
        if idx_rev in selected:
            ij_pairs.append((idx0, idx_rev))
            processed.add(idx0)
            processed.add(idx_rev)
        else:
            ij_pairs.append((idx0, None))
            processed.add(idx0)

    grouped = ij_pairs
    if len(grouped) == 0:
        return dict(
            LA_prop=LA,
            JD=0,
            acc_rate=0.0,
            thinned_k_size=0,
            total_k_size=0,
            n_eval=0,
            prob_prop=0.0,
            rev_prob_prop=0.0,
            prod_bal_con=0.0,
            rev_prod_bal_con=0.0,
        )

    omega_thin = float(np.clip(thinning_rate, 0.0, 1.0))
    total_k_size = int(len(grouped))
    n_eval = 0

    M = _get_moves()
    JD = 0

    prob_prop_log = 0.0
    rev_prob_prop_log = 0.0
    prod_bal_con_log = 0.0
    rev_prod_bal_con_log = 0.0
    G_curr = LA.curr.copy()
    for idx in range(total_k_size):
        # neighborhood thinning: with prob (1-omega) skip evaluating this sub-neighborhood
        if rng.random() >= omega_thin:
            continue
        n_eval += 1

        kj, kj_swap = grouped[idx]
        kj = int(kj)
        has_swap = kj_swap is not None
        if has_swap:
            kj_swap = int(kj_swap)

        i, j = (kj % d), (kj // d)
        temp_kj = int(G_curr[i, j])

        if has_swap:
            i2, j2 = (kj_swap % d), (kj_swap // d)
            temp_kj_swap = int(G_curr[i2, j2])

        # If this group is a singleton, only 2 distinct states (keep/flip).
        moves = M if has_swap else M[:2]
        n_moves = int(moves.shape[0])

        LA_cands: List[Optional[LAState]] = [None] * n_moves
        L_move = np.full(n_moves, -np.inf, dtype=float)

        # keep move
        LA_cands[0] = LA_temp
        L_move[0] = 0.0

        for m_idx in range(1, n_moves):
            di, dj = moves[m_idx]

            if di != 0:
                G_curr[i, j] = 1 - temp_kj
            if has_swap and dj != 0:
                G_curr[i2, j2] = 1 - temp_kj_swap

            # forbid immediate 2-cycle i<->j (paper uses DAG constraint anyway; keep this fast check)
            if (G_curr[i, j] == 1 and G_curr[j, i] == 1) or (
                has_swap and (G_curr[i2, j2] == 1 and G_curr[j2, i2] == 1)
            ):
                if di != 0: G_curr[i, j] = temp_kj
                if has_swap and dj != 0: G_curr[i2, j2] = temp_kj_swap
                continue
            
            added_edges = []
            if di != 0 and G_curr[i, j] == 1:
                added_edges.append((i, j))
            if has_swap and dj != 0 and G_curr[i2, j2] == 1:
                added_edges.append((i2, j2))
            is_valid_dag = True
            for (u, v) in added_edges:
                if has_path(G_curr, v, u):
                    is_valid_dag = False
                    break
            if not is_valid_dag:
                if di != 0:
                    G_curr[i, j] = temp_kj
                if has_swap and dj != 0:
                    G_curr[i2, j2] = temp_kj_swap
                continue
            # incremental update: only affected columns (child indices)
            cols = []
            if di != 0: cols.append(j + 1)
            if has_swap and dj != 0: cols.append(j2 + 1)
            if len(cols) == 0:
                # Revert just in case
                if di != 0: G_curr[i, j] = temp_kj
                if has_swap and dj != 0: G_curr[i2, j2] = temp_kj_swap
                LA_prop = LA_temp
                log_ok = 0.0
                LA_cands[m_idx] = LA_prop
                L_move[m_idx] = 0.0
                continue

            LA_prop = LAState(curr=G_curr, p_gam=int(G_curr.sum()))
            changes = np.unique(np.array(cols, dtype=int))
            LA_prop = log_llh_update(changes, LA_temp, LA_prop, hyper_par)
            lmp_prop = log_m_prior(LA_prop.p_gam, h, max_p)
            LA_prop.lmp = float(lmp_prop)
            LA_prop.log_post = float(LA_prop.llh + LA_prop.lmp)
            LA_cands[m_idx] = LA_prop

            # log_ok term (paper §3.1, matches the R implementation's k-prob ratio contribution)
            log_ok = 0.0
            if di != 0:
                log_ok += (2 * temp_kj - 1) * _safe_log_odds(float(PIPs[i, j]))
            if has_swap and dj != 0:
                log_ok += (2 * temp_kj_swap - 1) * _safe_log_odds(float(PIPs[i2, j2]))
            L_move[m_idx] = float(LA_prop.log_post - log_post_temp + log_ok)
            if di != 0:
                G_curr[i, j] = temp_kj
            if has_swap and dj != 0:
                G_curr[i2, j2] = temp_kj_swap

        # g(x)=min(1,x): log_g(L) = min(0, L)
        log_w = np.minimum(0.0, L_move)
        bal_const_log = _logsumexp(log_w)

        probs = np.exp(log_w - bal_const_log)
        u = rng.random()
        cum = np.cumsum(probs)
        chosen = int(np.searchsorted(cum, u, side="right"))
        if chosen >= n_moves:
            chosen = n_moves - 1
        if LA_cands[chosen] is None:
            chosen = 0

        if chosen != 0:
            di, dj = moves[chosen]
            if di != 0:
                G_curr[i, j] = 1 - temp_kj
            if has_swap and dj != 0:
                G_curr[i2, j2] = 1 - temp_kj_swap
        LA_temp = LA_cands[chosen]
        log_post_temp = float(LA_temp.log_post)

        JD += int(np.sum(moves[chosen]))
        prob_prop_log += float(log_w[chosen])
        prod_bal_con_log += float(-bal_const_log)

        # reverse weights relative to chosen state
        L_rel = L_move - L_move[chosen]
        rev_log_w = np.minimum(0.0, L_rel)
        rev_bal_const_log = _logsumexp(rev_log_w)

        rev_prob_prop_log += float(rev_log_w[0])          # probability to go back to base (keep)
        rev_prod_bal_con_log += float(-rev_bal_const_log)

    return dict(
        LA_prop=LA_temp,
        JD=int(JD),
        acc_rate=1.0,
        thinned_k_size=int(n_eval),
        total_k_size=int(total_k_size),
        n_eval=int(n_eval),
        prob_prop=float(prob_prop_log),
        rev_prob_prop=float(rev_prob_prop_log),
        prod_bal_con=float(prod_bal_con_log),
        rev_prod_bal_con=float(rev_prod_bal_con_log),
    )

# =============================================================================
# 5) Omega thinning (paper §3.4 Eq.(13)) and PIPs adaptation (paper §3.2 Eq.(9))
# =============================================================================
def logit_e(x: np.ndarray, eps: float) -> np.ndarray:
    x = x.copy()
    x[x > 1 - 2 * eps] = 1 - 2 * eps
    x[x < 2 * eps] = 2 * eps
    return np.log(x - eps) - np.log(1 - x - eps)

def inv_logit_e(y: np.ndarray, eps: float) -> np.ndarray:
    ey = np.exp(-y)
    return (eps * ey - eps + 1) / (ey + 1)

def _logit_e_scalar(x: float, eps: float) -> float:
    return float(logit_e(np.array([float(x)], dtype=float), eps)[0])

def _inv_logit_e_scalar(y: float, eps: float) -> float:
    return float(inv_logit_e(np.array([float(y)], dtype=float), eps)[0])

def _omega_robbins_monro_update(ctx: dict, Nt: int) -> None:
    """Paper §3.4 Eq.(13) Robbins–Monro update for omega thinning."""
    if not bool(ctx.get("omega_adapt", False)):
        return
    try:
        N_tilde = float(ctx.get("omega_N_tilde", 10.0))
    except Exception:
        return
    if (not np.isfinite(N_tilde)) or (N_tilde <= 0):
        return

    eps = float(ctx.get("omega_eps", 1e-6))
    gamma = float(ctx.get("omega_rm_gamma", 0.7))
    psi_scale = float(ctx.get("omega_rm_psi_scale", 1.0))
    t = int(ctx.get("omega_t", 0)) + 1
    psi_t = psi_scale * (t ** (-gamma))

    omega = float(ctx.get("omega_thin", ctx.get("omega", 1.0)))
    omega = float(np.clip(omega, 0.0, 1.0))
    logit_omega = _logit_e_scalar(omega, eps)

    logit_next = float(logit_omega - psi_t * (float(Nt) - N_tilde))
    omega_next = float(_inv_logit_e_scalar(logit_next, eps))
    omega_next = float(np.clip(omega_next, 0.0, 1.0))

    ctx["omega_t"] = t
    ctx["omega_logit"] = logit_next
    ctx["omega_thin"] = omega_next
    ctx["omega"] = omega_next
    ctx["omega_last_Nt"] = int(Nt)
    ctx["omega_last_target"] = float(N_tilde)
    ctx["omega_last_psi"] = float(psi_t)

def _phi_schedule_S8(t: int, Nb: int) -> float:
    """
    Appendix B Eq.(S.8):
      φ_t = 1 - 1/2 * (1/(Nb - t + 1))^0.2,  if t <= Nb
          = 1/2 * (1/(t - Nb))^0.5,          if t > Nb
    """
    t = int(max(1, t))
    Nb = int(max(0, Nb))
    if Nb > 0 and t <= Nb:
        denom = float(Nb - t + 1)
        return float(1.0 - 0.5 * (1.0 / denom) ** 0.2)
    denom = float(max(1, t - Nb))
    return float(0.5 * (1.0 / denom) ** 0.5)

def _recompute_AD_from_PIPs(PIPs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    A_ij = min(P/(1-P), 1), D_ij = min((1-P)/P, 1)
    (matches the paper's behavior example in §3.2 and your original parni_dag_2 implementation)
    """
    P = np.asarray(PIPs, dtype=float)
    A = np.minimum(P / np.maximum(1e-12, 1.0 - P), 1.0)
    D = np.minimum((1.0 - P) / np.maximum(1e-12, P), 1.0)
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(D, 0.0)
    return A, D

def parni_update_pips_eq9(ctx: dict, G_curr: np.ndarray) -> dict:
    """
    Paper §3.2 Eq.(9) adaptive update for η / PIPs.
    Call AFTER outer MH accept/reject, using the chain's current graph γ^{(t)}.
    """
    if not bool(ctx.get("pips_adapt", True)):
        return {"enabled": False}

    t = int(ctx.get("pips_t", 0)) + 1
    ctx["pips_t"] = t
    Nb = int(ctx.get("pips_Nb", 0))

    if "pi_tilde" not in ctx or ctx["pi_tilde"] is None:
        ctx["pi_tilde"] = np.array(ctx["PIPs"], copy=True, dtype=float)
    pi_tilde = np.asarray(ctx["pi_tilde"], dtype=float)

    if "pi_hat" not in ctx or ctx["pi_hat"] is None:
        ctx["pi_hat"] = np.zeros_like(pi_tilde, dtype=float)
    pi_hat = np.asarray(ctx["pi_hat"], dtype=float)

    gamma = np.asarray(G_curr, dtype=float)
    np.fill_diagonal(gamma, 0.0)

    pi_hat += (gamma - pi_hat) / float(t)
    ctx["pi_hat"] = pi_hat

    phi_t = _phi_schedule_S8(t, Nb)
    ctx["phi_t"] = float(phi_t)

    eta_hat = phi_t * pi_tilde + (1.0 - phi_t) * pi_hat

    eps = float(ctx.get("pips_eps", 1e-3))
    eta_hat = np.clip(eta_hat, eps, 1.0 - eps)
    np.fill_diagonal(eta_hat, 0.0)

    ctx["PIPs"] = eta_hat
    A, D = _recompute_AD_from_PIPs(eta_hat)
    ctx["A"] = A
    ctx["D"] = D

    mask = ~np.eye(eta_hat.shape[0], dtype=bool)
    return {
        "enabled": True,
        "t": int(t),
        "Nb": int(Nb),
        "phi_t": float(phi_t),
        "mean_pips": float(np.mean(eta_hat[mask])),
        "min_pips": float(np.min(eta_hat[mask])),
        "max_pips": float(np.max(eta_hat[mask])),
    }


# =============================================================================
# 5.5) Paper §3.2 warm-start: approximate oriented PIPs from skeleton H (BGe)
# =============================================================================
def _local_log_prior_size(s: int, m: int, hval) -> float:
    """Local prior over parent-set size s (m = p-1). Supports Bernoulli(h) and Beta-Binomial(a,b).
    Note: constants that do not depend on s are dropped (as in the original code) since we only
    need relative weights for warm-start enumeration.
    """
    if isinstance(hval, (tuple, list)):
        a, b = float(hval[0]), float(hval[1])
        # log Beta(a+s, b+(m-s)) - log Beta(a, b)  (up to constants)
        return float(math.lgamma(a + s) + math.lgamma(b + (m - s)) - math.lgamma(a + b + m))
    h = float(hval)
    h = min(max(h, 1e-12), 1.0 - 1e-12)
    return float(s * math.log(h) + (m - s) * math.log(1.0 - h))

def _iter_subsets(cand_idx, max_enum: Optional[int] = None):
    """Enumerate subsets of cand_idx. If 2^m exceeds max_enum, enumerate all small subsets first,
    then randomly sample additional subsets to reach budget (paper-inspired practical guard).
    """
    cand_idx = list(int(x) for x in cand_idx)
    m = len(cand_idx)
    if max_enum is None or (2 ** m) <= int(max_enum):
        from itertools import combinations
        for k in range(m + 1):
            for comb in combinations(cand_idx, k):
                yield np.array(comb, dtype=int)
    else:
        from itertools import combinations
        budget = int(max_enum)
        count = 0
        # fully enumerate small subsets first (keeps the most important mass in sparse regime)
        for k in range(min(m, 3) + 1):
            for comb in combinations(cand_idx, k):
                yield np.array(comb, dtype=int)
                count += 1
                if count >= budget:
                    return
        rng = np.random.default_rng(0)
        while count < budget:
            k = int(rng.integers(low=0, high=min(m, 6) + 1))
            if k == 0:
                yield np.array([], dtype=int)
            else:
                S = rng.choice(cand_idx, size=k, replace=False)
                yield np.sort(S).astype(int)
            count += 1

def paper_marPIPs_from_H(
    hp: HyperPar,
    H: np.ndarray,
    kappa: float = 0.0,
    extend_one: bool = True,
    max_enum_parents: int = 8,
    max_enum_sets: int = 4096,
    extra_parents: Optional[Dict[int, List[int]]] = None,
) -> np.ndarray:
    """Paper §3.2 warm-start: approximate oriented edge PIPs from a skeleton H using BGe local scores.

    This is a faithful, *practical* version:
      - per-node parent-set enumeration restricted by H[:, j]
      - optional one-step expansion (h_j^+) by adding 1 strong outside candidate
      - truncate very large candidate sets to max_enum_parents using |XtX| heuristic
      - convert marginal (parent inclusion) probabilities to oriented edge probabilities with mutual exclusion
      - apply kappa shrinkage toward 0.5 to avoid extreme PIPs

    Returns:
      PIPs (p,p) oriented, diagonal = 0.
    """
    p = int(hp.p)
    H = np.asarray(H, dtype=int)
    if H.shape != (p, p):
        raise ValueError(f"H must have shape {(p, p)}, got {H.shape}")
    H = (H != 0).astype(int)
    np.fill_diagonal(H, 0)

    # We rely on BGeAdapter for local scores.
    adapter: BGEAdapter = hp.tables
    if adapter is None:
        raise ValueError("paper_marPIPs_from_H requires hp.tables (BGEAdapter).")

    # correlation / relevance heuristic for truncation
    XtX = hp.XtX
    if XtX is None:
        X = np.asarray(hp.X, dtype=float)
        XtX = X @ X.T
    XtX = np.asarray(XtX, dtype=float)

    P_raw = np.zeros((p, p), dtype=float)  # marginal parent inclusion (not oriented yet)

    for j in range(p):
        cand = np.where(H[:, j] == 1)[0]
        cand = cand[cand != j]
        if cand.size == 0:
            continue

        # truncate too-large candidate parent sets by absolute cross-correlation
        if cand.size > int(max_enum_parents):
            wj = np.abs(XtX[cand, j])
            take = np.argsort(-wj)[: int(max_enum_parents)]
            cand = cand[take]

        subsets = list(_iter_subsets(cand, max_enum=int(max_enum_sets)))

        # optional h_j^+ expansion: add 1 outside candidate to compare (paper)
        if extend_one:
            outside = np.setdiff1d(np.arange(p), np.append(cand, j))
            if outside.size > 0:
                # Appendix C: h_j^+ uses one extra parent candidate outside skeleton.
                # Prefer user-provided extra_parents[j] if available; otherwise fall back to XtX heuristic.
                keep_o = None
                if extra_parents is not None and int(j) in extra_parents and extra_parents[int(j)]:
                    cand_list = [int(o) for o in extra_parents[int(j)] if int(o) != int(j)]
                    # intersect with outside set
                    outside_set = set(int(x) for x in outside.tolist())
                    cand_list = [o for o in cand_list if o in outside_set]
                    if cand_list:
                        keep_o = np.array(cand_list[: min(10, len(cand_list))], dtype=int)
                if keep_o is None:
                    wout = np.abs(XtX[outside, j])
                    keep_o = outside[np.argsort(-wout)[: min(10, outside.size)]]
                ext_sets = []
                for S in subsets:
                    for o in keep_o:
                        ext_sets.append(np.sort(np.append(S, o)))
                if ext_sets:
                    # unique
                    arr_tuples = list(dict.fromkeys(tuple(int(x) for x in s) for s in ext_sets))
                    subsets += [np.array(t, dtype=int) for t in arr_tuples]
                    if len(subsets) > int(max_enum_sets):
                        subsets = subsets[: int(max_enum_sets)]

        m = p - 1
        Ws = []
        for S in subsets:
            # BGe local log marginal likelihood for node j with parent set S
            parents_vec = np.zeros(p, dtype=int)
            if len(S) > 0:
                parents_vec[np.asarray(S, dtype=int)] = 1
            ll = float(adapter.bge._mll_per_variable(int(j), parents_vec, adapter.R, adapter.N))
            lp = _local_log_prior_size(int(len(S)), m, hp.h)
            Ws.append(ll + lp)
        Ws = np.asarray(Ws, dtype=float)
        Ws = Ws - float(np.max(Ws))
        w = np.exp(Ws)
        Z = float(np.sum(w))
        if (not np.isfinite(Z)) or Z <= 0:
            continue
        post = w / Z

        for S, prob in zip(subsets, post):
            if len(S) > 0:
                P_raw[np.asarray(S, dtype=int), j] += float(prob)

    # mutual exclusion normalization: forbid i<->j simultaneously
    P = P_raw.copy()
    eps = 1e-9
    odds = np.clip(P / np.clip(1.0 - P, eps, None), eps, 1.0 / eps)
    for i in range(p):
        for j in range(i + 1, p):
            oij, oji = float(odds[i, j]), float(odds[j, i])
            denom = 1.0 + oij + oji
            P[i, j] = oij / denom
            P[j, i] = oji / denom
    np.fill_diagonal(P, 0.0)

    # kappa shrinkage toward 0.5 (avoid extreme PIPs)
    if kappa and float(kappa) > 0:
        kk = float(kappa)
        P = (1.0 - kk) * P + kk * 0.5
        np.fill_diagonal(P, 0.0)

    return P
# =============================================================================
# 6) Public interface for long_multilevel_splitting_frame
# =============================================================================
def parni_prepare_context(
    X_p_n: np.ndarray,
    h,
    bge_obj: Optional[BGe] = None,
    H: Optional[np.ndarray] = None,      # kept for signature compatibility (unused in minimal build)
    kappa: float = 0.0,                  # kept for signature compatibility (unused)
    omega: float = 1.0,
    pips_mode: str = "uniform",          # "uniform" (with H) or "bge" warm-start
    pips_in: float = 0.25,                   # PIPs value for edges within H when pips_mode="uniform"
    pips_out: float = 0.02,                  # exploration PIPs for edges outside H
    extra_parents: Optional[Dict[int, List[int]]] = None,  # Appendix C h_j^+ outside candidates
) -> dict:
    """
    Build a minimal PARNI context. For simplicity, we always initialize PIPs uniformly at 0.5
    (except diagonal), which is sufficient for correctness; warm-start enumeration from H is removed.
    """
    X_p_n = np.asarray(X_p_n, dtype=float)
    p, n = X_p_n.shape
    if bge_obj is None:
        raise ValueError("Minimal parni_dag requires bge_obj (BGe) to be provided.")

    adapter = BGEAdapter(bge_obj, X_p_n)

    # Precompute cross-product for warm-start heuristics (used by paper_marPIPs_from_H)
    XtX = X_p_n @ X_p_n.T

    hp = HyperPar(
        X=X_p_n,
        XtX=XtX,
        h=h,
        p=int(p),
        n=int(n),
        max_p=int(p * (p - 1)),
        log_llh=log_llh_BGE,
        log_llh_update=log_llh_BGE_update_table,
        log_m_prior=lambda pg, hh, mp: _log_m_prior_beta_binom(int(pg), hh, int(mp)),
        tables=adapter,
    )

    # Initialize PIPs
    pips_mode_req = str(pips_mode).lower()
    if pips_mode_req == "bge":
        # Use paper §3.2 warm-start from skeleton H and BGe local scores
        if H is None:
            H_eff = np.ones((p, p), dtype=int) - np.eye(p, dtype=int)
        else:
            H_eff = np.asarray(H, dtype=int)
            if H_eff.shape != (p, p):
                raise ValueError(f"H must have shape {(p, p)}, got {H_eff.shape}")
            H_eff = (H_eff != 0).astype(int)
            np.fill_diagonal(H_eff, 0)
        PIPs = paper_marPIPs_from_H(
            hp,
            H_eff,
            kappa=float(kappa),
            extend_one=True,
            max_enum_parents=min(8, p - 1),
            max_enum_sets=4096,
            extra_parents=extra_parents,
        )
        pips_mode_used = "bge"
    else:
        pips_mode_used = "uniform"
        if H is None:
            PIPs = np.full((hp.p, hp.p), 0.5, float)
            np.fill_diagonal(PIPs, 0.0)
        else:
            Hm = (np.asarray(H) != 0)
            Hm = np.maximum(Hm, Hm.T)
            PIPs = np.full((hp.p, hp.p), float(pips_out), float)
            p_in = float(pips_in)
            if not (0.0 < p_in < 1.0):
                raise ValueError(f"pips_in must be in (0,1), got {p_in}")
            # Cap at 0.5 to remain "uniform-ish" and keep A/D bounded.
            p_in = min(p_in, 0.5)
            PIPs[Hm] = p_in
            np.fill_diagonal(PIPs, 0.0)

    A, D = _recompute_AD_from_PIPs(PIPs)

    ctx = dict(
        hp=hp,
        PIPs=np.array(PIPs, copy=True, dtype=float),
        A=A,
        D=D,
        # Eq.(9) state
        pi_tilde=np.array(PIPs, copy=True, dtype=float),
        pi_hat=np.zeros_like(PIPs, dtype=float),
        pips_adapt=True,
        pips_t=0,
        pips_Nb=0,
        pips_eps=1e-3,
        # omega thinning
        omega=float(omega),
        omega_thin=float(omega),
        omega_adapt=False,       # paper feature; default off in minimal build
        omega_t=0,
        omega_eps=1e-6,
        omega_rm_gamma=0.7,
        omega_rm_psi_scale=1.0,
        omega_N_tilde=10.0,
        # Hastings balancing function (paper recommends g(x)=min(1,x))
        bal_fun=(lambda x: min(1.0, float(x))),
        # bookkeeping
        pips_mode_requested=str(pips_mode),
        pips_mode_used=str(pips_mode_used),
    )
    return ctx

def parni_make_LA_from_G(G: np.ndarray, ctx: dict) -> LAState:
    return compute_LA_DAG(np.asarray(G, dtype=int), ctx["hp"])

def parni_step_one(
    LA: LAState,
    ctx: dict,
    rng: Optional[np.random.Generator] = None,
    proposal_only: bool = False,
):
    """
    One PARNI structure step.
      - proposal_only=False: returns (LA_new, accepted, info_dict)
      - proposal_only=True:  returns dict with LA_prop, log_qG_fwd, log_qG_rev, ...
    """
    if rng is None:
        rng = np.random.default_rng()

    hp = ctx["hp"]
    A = ctx["A"]
    D = ctx["D"]
    PIPs = ctx["PIPs"]
    gfun = ctx["bal_fun"]

    # η_ij = (1-γ_ij)A_ij + γ_ij D_ij
    eta = (1 - LA.curr) * A + LA.curr * D

    neigh = sample_ind_DAG(True, eta, None, log=True, rng=rng)
    k = neigh["sample"]

    if k is None or (isinstance(k, np.ndarray) and k.size == 0):
        if proposal_only:
            return {
                "proposal_only": True,
                "LA_prop": LA,
                "log_qG_fwd": 0.0,
                "log_qG_rev": 0.0,
                "k_raw_size": 0,
                "k_total_groups": 0,
                "n_eval": 0,
                "k_size": 0,
                "log_post_curr": float(LA.log_post),
                "log_post_prop": float(LA.log_post),
                "omega_thin_before": float(ctx.get("omega_thin", ctx.get("omega", 1.0))),
                "omega_thin_after": float(ctx.get("omega_thin", ctx.get("omega", 1.0))),
                "omega_t": int(ctx.get("omega_t", 0)),
                "reason": "empty_k",
            }
        return LA, False, {"reason": "empty_k"}

    omega_thin_before = float(ctx.get("omega_thin", ctx.get("omega", 1.0)))

    upd = update_LA_DAG(LA, k, hp, gfun, PIPs, thinning_rate=omega_thin_before, rng=rng)

    Nt = int(upd.get("n_eval", upd.get("thinned_k_size", 0)))
    _omega_robbins_monro_update(ctx, Nt)

    omega_thin_after = float(ctx.get("omega_thin", omega_thin_before))
    LA_prop: LAState = upd["LA_prop"]

    log_qG_fwd = float(upd["prob_prop"] + upd["prod_bal_con"])
    log_qG_rev = float(upd["rev_prob_prop"] + upd["rev_prod_bal_con"])

    if proposal_only:
        return {
            "proposal_only": True,
            "LA_prop": LA_prop,
            "log_qG_fwd": log_qG_fwd,
            "log_qG_rev": log_qG_rev,
            "k_raw_size": int(np.size(k)),
            "k_total_groups": int(upd.get("total_k_size", 0)),
            "n_eval": int(upd.get("n_eval", upd.get("thinned_k_size", 0))),
            "k_size": int(upd.get("thinned_k_size", 0)),
            "log_post_curr": float(LA.log_post),
            "log_post_prop": float(LA_prop.log_post),
            "omega_thin_before": float(omega_thin_before),
            "omega_thin_after": float(omega_thin_after),
            "omega_t": int(ctx.get("omega_t", 0)),
        }

    log_alpha = (
        float(LA_prop.log_post - LA.log_post)
        + float(upd["rev_prob_prop"] - upd["prob_prop"])
        + float(upd["rev_prod_bal_con"] - upd["prod_bal_con"])
    )
    accept = (np.log(rng.random()) < log_alpha)
    return (LA_prop if accept else LA), bool(accept), {
        "log_alpha": float(log_alpha),
        "k_raw_size": int(np.size(k)),
        "k_total_groups": int(upd.get("total_k_size", 0)),
        "n_eval": int(upd.get("n_eval", upd.get("thinned_k_size", 0))),
        "k_size": int(upd.get("thinned_k_size", 0)),
        "accepted": bool(accept),
        "omega_thin_before": float(omega_thin_before),
        "omega_thin_after": float(omega_thin_after),
        "omega_t": int(ctx.get("omega_t", 0)),
    }