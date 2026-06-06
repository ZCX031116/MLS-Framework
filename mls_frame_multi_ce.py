import os
import sys
import math
import random
import shutil
import tempfile
import warnings
import hashlib
from pathlib import Path
from itertools import permutations
from multiprocessing import Manager
from copy import deepcopy
import copy
import numpy as np
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from adjustText import adjust_text
from collections import defaultdict, deque
from adjustText import adjust_text
from dataclasses import dataclass
import json
from src.bge import BGe
from src.PC_skeleton import build_H_pc_plus
from src.parni_dag import(
    parni_prepare_context,
    parni_make_LA_from_G,
    parni_step_one,
    parni_update_pips_eq9,
)
from src.helper_func import (
    pairwise_linear_ce, 
    ce_ij, 
    pairwise_linear_ce_no_params, 
    log_and_print,
    plot_edge_frequency_and_weight_avg_graphs,
    get_erdos_renyi_q,
    get_p_edge_for_inference,
    build_H_topk_corr,
    p_structure_schedule,
    check_acyclic,
    build_reachability,
    level_tag,
    atomic_json_dump,
    summarize_array,
    draw_graphs_in_grid,
    sample_random_graphs,
    unique_graph_count,
    sanity_check_weights_matrix
)

class Multilevel:
    def __init__(self, bge_model, data, X, ce_constraints, save_dir, output_file, max_outer_iter,
                    edges_per_node = 2, params_per_graph=50, rng=None, structure_kernel = "Structure_MCMC", p_structure = 0.5,
                    save_level_samples: bool = True, save_level_weights: bool = True, save_level_all: bool = False,
                    samples_dirname = "level_samples"):
        d = data.shape[1]
        self.bge_model = bge_model
        self.R = bge_model.calc_R(data)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.data = data
        # --- multi-CE only ---
        # ce_constraints: list of (i, j, op, threshold, scale), op in {">","<"}
        self.ce_constraints = self._prepare_ce_constraints(ce_constraints)
        self._ce_signs = np.asarray([c["sign"] for c in self.ce_constraints], dtype=float)
        self._ce_thr_metric = np.asarray([c["thr_metric"] for c in self.ce_constraints], dtype=float)
        self._ce_scales = np.asarray([c["scale"] for c in self.ce_constraints], dtype=float)
        self.X = self._prepare_score_targets(X)
        # primary pair (for logging / sanity diagnostics)
        self.i = int(self.ce_constraints[0]["i"])
        self.j = int(self.ce_constraints[0]["j"])
        self.sigma = 1.0
        self.mu = 0.0
        self.save_dir = save_dir
        self.output_file = output_file
        self.p_edge = get_p_edge_for_inference(d, edges_per_node)
        self.params_per_graph = params_per_graph
        self.max_outer_iter=max_outer_iter
        self.structure_kernel = structure_kernel
        self.p_structure = p_structure
        self.save_level_samples = bool(save_level_samples)
        self.save_level_weights = bool(save_level_weights)
        self.save_level_all = bool(save_level_all)
        self.samples_dirname = str(samples_dirname)
        self.level_samples_dir = Path(self.save_dir) / self.samples_dirname
        if self.save_level_samples:
            self.level_samples_dir.mkdir(parents=True, exist_ok=True)
        self.enable_extra_sanity = True
        self.sanity_topk_edges = 8   
        self.sanity_eps = 1e-12   

        if self.structure_kernel == "PARNI":
            print("Preparing PARNI context...")
            X_p_n = self.data.T
            d = self.data.shape[1]
            K = min(8, d - 1)
            H, extra_parents = build_H_pc_plus(
                self.data,
                alpha=0.1,
                max_cond_set=6,
                candidate_cap=12,
                extend_one=True,
                extra_parent_cap=8,
                verbose=False,
            )
            pips_in = 0.5
            pips_out = 0.5
            self.parni_ctx = parni_prepare_context(
                X_p_n=X_p_n,
                h=self.p_edge,
                bge_obj=self.bge_model,
                H=H,
                kappa=0.1,      
                omega=0.5,       
                pips_mode="uniform",
                pips_in=pips_in,
                pips_out=pips_out, 
                extra_parents=extra_parents,
            )
            self.parni_ctx["omega_N_tilde"] = 10 if d<=16 else 20          
            self.parni_ctx["omega_adapt"] = True
            self.parni_ctx["pips_adapt"] = False
            self.parni_ctx["pips_eps"] = 0.05


    # ------------------------------------------------------------------
    # Multi-constraint CE conditioning helpers (multi-CE only version)
    # ce_constraints format (fixed):
    #   List[Tuple[int,int,str,float,float]] where each entry is
    #     (i, j, op, threshold, scale)
    #   op in {">", "<"} represents CE(i->j) > threshold  or  CE(i->j) < threshold.
    #
    # Splitting score:
    #   score(W) = min_k  ( (sign_k * CE_k(W) - sign_k * thr_k) / scale_k )
    # where sign_k = +1 for ">", -1 for "<".
    #
    # Base conjunction event is score >= 0.
    # Therefore X is interpreted in score-space (e.g. X=[0.0]).
    # ------------------------------------------------------------------

    def _prepare_score_targets(self, X):
        if X is None:
            return [0.0]
        vals = [float(v) for v in list(X)]
        return sorted(vals)

    def _prepare_ce_constraints(self, ce_constraints):
        if ce_constraints is None:
            raise ValueError("ce_constraints must be provided (multi-CE only version).")
        if (not isinstance(ce_constraints, (list, tuple))) or len(ce_constraints) == 0:
            raise ValueError("ce_constraints must be a non-empty list of 5-tuples: (i,j,op,threshold,scale).")

        out = []
        for idx, spec in enumerate(list(ce_constraints)):
            if (not isinstance(spec, (list, tuple))) or len(spec) != 5:
                raise ValueError(
                    f"Constraint #{idx} must be a 5-tuple (i,j,op,threshold,scale). Got: {spec}"
                )
            i, j, op, thr, scale = spec
            op = str(op).strip()
            if op not in (">", "<"):
                raise ValueError(f"Constraint #{idx} op must be '>' or '<'. Got: {op!r}")
            i = int(i)
            j = int(j)
            thr = float(thr)
            scale = float(scale)
            if (not np.isfinite(scale)) or (scale <= 0.0):
                raise ValueError(f"Constraint #{idx} scale must be positive finite. Got: {scale}")
            sign = 1.0 if op == ">" else -1.0
            out.append(
                {"i": i, "j": j, "op": op, "thr": thr, "scale": scale, "sign": sign, "thr_metric": sign * thr}
            )
        return out

    def _splitting_metric_details(self, weight_matrix):
        raw_vec = np.asarray(
            [float(ce_ij(weight_matrix, c["i"], c["j"])) for c in self.ce_constraints],
            dtype=float,
        )
        metric_vec = self._ce_signs * raw_vec
        margin_vec = (metric_vec - self._ce_thr_metric) / self._ce_scales
        score = float(np.min(margin_vec))
        return score, raw_vec, metric_vec, margin_vec

    def _splitting_metric(self, weight_matrix):
        return self._splitting_metric_details(weight_matrix)[0]

    def _condition_str(self, score_level, with_indices=True):
        L = float(score_level)
        parts = []
        for c in self.ce_constraints:
            i = int(c["i"])
            j = int(c["j"])
            op = c["op"]
            thr = float(c["thr"])
            s = float(c["scale"])
            pair = f"CE[{i},{j}]" if with_indices else "CE"
            if op == ">":
                thr_shift = thr + L * s
                parts.append(f"{pair} > {thr_shift:.4g}")
            else:
                thr_shift = thr - L * s
                parts.append(f"{pair} < {thr_shift:.4g}")
        return " && ".join(parts)


    def _clone_parni_ctx_for_chain(self, base_ctx, burn_in = None):
        ctx = dict(base_ctx)  # shallow copy
        if "hp" in base_ctx:
            ctx["hp"] = base_ctx["hp"]
        if "bal_fun" in base_ctx:
            ctx["bal_fun"] = base_ctx["bal_fun"]
        for k in ("PIPs", "A", "D", "pi_tilde", "pi_hat"):
            if k in base_ctx and base_ctx[k] is not None:
                ctx[k] = np.array(base_ctx[k], copy=True)
        ctx["pips_t"] = 0
        if "pi_hat" in ctx and ctx["pi_hat"] is not None:
            ctx["pi_hat"] = np.zeros_like(ctx["pi_hat"], dtype=float)
        if "pi_tilde" in ctx and ctx["pi_tilde"] is not None:
            ctx["PIPs"] = np.array(ctx["pi_tilde"], copy=True)
            A, D = parni._recompute_AD_from_PIPs(ctx["PIPs"])
            ctx["A"], ctx["D"] = A, D
        omega0 = float(base_ctx.get("omega", base_ctx.get("omega_thin", 1.0)))
        ctx["omega"] = float(omega0)
        ctx["omega_thin"] = float(omega0)
        ctx["omega_t"] = 0
        ctx.pop("omega_logit", None)
        ctx.pop("omega_last_Nt", None)
        ctx.pop("omega_last_target", None)
        ctx.pop("omega_last_psi", None)
        if burn_in is not None:
            ctx["pips_Nb"] = max(1, int(burn_in))
        return ctx
            
    def log_post_over_weights(self, adjacency_matrix, weight_matrix, bge_model, data):
        log_p = 0.0
        N, d = data.shape
        R = self.R
        alpha_w = bge_model.alpha_w
        alpha_w_prime = alpha_w + N 
        for i in range(d):
            parents = np.where(adjacency_matrix[:, i])[0].tolist()
            l = len(parents) + 1 
            if not parents:
                if np.any(weight_matrix[:, i] != 0):
                    return -np.inf 
                else:
                    continue
            R11 = R[np.ix_(parents, parents)]
            R12 = R[parents, i]
            R21 = R[i, parents]
            R22 = R[i, i]
            try:
                R11_inv = np.linalg.inv(R11)
            except np.linalg.LinAlgError:
                return -np.inf 
            loc = R11_inv @ R12
            denom = R22 - R21 @ R11_inv @ R12
            if denom <= 1e-10:  
                return -np.inf
            deg_free = alpha_w_prime - d + l
            shape_matrix = np.linalg.inv((deg_free / denom) * R11)
            eigenvalues = np.linalg.eigvalsh(shape_matrix)
            b_i = weight_matrix[parents, i]
            try:
                dist = st.multivariate_t(loc=loc, shape=shape_matrix, df=deg_free)
                log_p += dist.logpdf(b_i)
            except np.linalg.LinAlgError:
                return -np.inf
        return log_p

    def log_posterior_with_weights(self, adjacency_matrix, weight_matrix, mll_score = None):
        num_edges = np.sum(adjacency_matrix)
        logit_p = np.log(self.p_edge / (1.0 - self.p_edge))
        prior = num_edges * logit_p 
        if np.isneginf(prior) or np.isnan(prior):
            prior = -100.0
        log_prior_g = prior
        if mll_score is None:
            mll_score = self.bge_model.mll(adjacency_matrix, self.data)
        log_post_w = self.log_post_over_weights(adjacency_matrix, weight_matrix, self.bge_model, self.data)
        return (log_prior_g + mll_score + log_post_w), log_prior_g, mll_score, log_post_w

    def score_state(self, adjacency_matrix, weight_matrix, pairwise_effect=None, level=0, mll_score = None):
        G_copy = np.array([np.copy(adjacency_matrix)])
        if pairwise_effect is None:
            pairwise_effect = self._splitting_metric(weight_matrix)
        if pairwise_effect < level:
            return -np.inf, pairwise_effect, -np.inf, -np.inf, -np.inf
        log_score_val,log_prior_g, mll_score,log_post_w = self.log_posterior_with_weights(
            adjacency_matrix, weight_matrix, 
            mll_score = mll_score
        )
        return log_score_val, pairwise_effect,log_post_w, log_prior_g, mll_score

    def initialize_edge_weight_matrix(self, adjacency_matrix):
        G_batch = np.array(adjacency_matrix)
        Bs, d_care = pairwise_linear_ce_no_params(G_batch, self.data, self.bge_model,
                                          params_per_graph=1, avg=False, return_B=True)
        return Bs

    def _node_weight_posterior_params(self, adj: np.ndarray, v: int):
        parents = np.where(adj[:, v])[0].tolist()
        if len(parents) == 0:
            return parents, None, None, None
        N, d = self.data.shape
        R = self.R
        alpha_w = self.bge_model.alpha_w
        alpha_w_prime = alpha_w + N
        l = len(parents) + 1
        try:
            R11 = R[np.ix_(parents, parents)]
            R12 = R[parents, v]
            R21 = R[v, parents]
            R22 = float(R[v, v])
            loc = np.linalg.solve(R11, R12)
            denom = float(R22 - R21 @ loc)
            if (not np.isfinite(denom)) or denom <= 1e-10:
                raise np.linalg.LinAlgError
            df = float(alpha_w_prime - d + l)
            if (not np.isfinite(df)) or df <= 0.0:
                raise np.linalg.LinAlgError
            invR11 = np.linalg.solve(R11, np.eye(len(parents)))
            shape = (denom / df) * invR11
            shape = 0.5 * (shape + shape.T)
            if not np.all(np.isfinite(shape)):
                raise np.linalg.LinAlgError
            return parents, np.asarray(loc, dtype=float), np.asarray(shape, dtype=float), df
        except np.linalg.LinAlgError:
            return parents, None, None, None

    def _node_weight_logpdf(self, adj: np.ndarray, v: int, b_vec: np.ndarray) -> float:
        """log q(b_vec) under the node-block proposal for node v (given adj)."""
        parents, loc, shape, df = self._node_weight_posterior_params(adj, v)
        if len(parents) == 0:
            return 0.0
        b_vec = np.atleast_1d(np.asarray(b_vec, dtype=float))
        if loc is None:
            return float(np.sum(st.norm.logpdf(b_vec, loc=self.mu, scale=self.sigma)))
        dist = st.multivariate_t(loc=loc, shape=shape, df=df)
        return float(dist.logpdf(b_vec))

    def _sample_node_weight_block(self, adj: np.ndarray, v: int, rng):
        parents, loc, shape, df = self._node_weight_posterior_params(adj, v)
        if len(parents) == 0:
            return parents, np.empty((0,), dtype=float)
        if loc is None:
            b_new = rng.normal(loc=self.mu, scale=self.sigma, size=len(parents))
            return parents, np.asarray(b_new, dtype=float)
        dist = st.multivariate_t(loc=loc, shape=shape, df=df)
        b_new = dist.rvs(random_state=rng)
        b_new = np.atleast_1d(np.asarray(b_new, dtype=float))
        return parents, b_new

    def _changed_nodes_by_parents(self, G: np.ndarray, Gp: np.ndarray):
        d = G.shape[0]
        changed = []
        for v in range(d):
            pa_G  = np.where(G[:, v])[0]
            pa_Gp = np.where(Gp[:, v])[0]
            if pa_G.shape[0] != pa_Gp.shape[0] or (pa_G.shape[0] > 0 and not np.array_equal(pa_G, pa_Gp)):
                changed.append(v)
        return changed

    def _refresh_weights_S1(self, G: np.ndarray, W: np.ndarray, Gp: np.ndarray, rng):
        G  = np.asarray(G, dtype=int)
        Gp = np.asarray(Gp, dtype=int)
        W  = np.asarray(W, dtype=float)
        Wp = np.array(W, copy=True)
        changed = self._changed_nodes_by_parents(G, Gp)
        for v in changed:
            pa_new, b_new = self._sample_node_weight_block(Gp, v, rng)
            Wp[:, v] = 0.0
            if len(pa_new) > 0:
                Wp[pa_new, v] = b_new
        Wp[~Gp.astype(bool)] = 0.0
        np.fill_diagonal(Wp, 0.0)
        return Wp, 0.0, 0.0, changed

    def propose_new_structure_only(self, adj: np.ndarray, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        A = np.asarray(adj, dtype=int)
        d = A.shape[0]
        def _has_path(src: int, dst: int, ignore_edge=None) -> bool:
            if src == dst:
                return True
            visited = np.zeros(d, dtype=bool)
            q = deque([src])
            visited[src] = True
            while q:
                u = q.popleft()
                nbrs = np.flatnonzero(A[u] != 0)
                for v in nbrs:
                    if ignore_edge is not None and u == ignore_edge[0] and v == ignore_edge[1]:
                        continue
                    if v == dst:
                        return True
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
            return False
        u = int(rng.integers(d))
        v = int(rng.integers(d - 1))
        if v >= u:
            v += 1
        if u > v:
            u, v = v, u  # ensure u < v
        uv = int(A[u, v] != 0)
        vu = int(A[v, u] != 0)
        # Pair state encoding: 0 = none, 1 = u->v, 2 = v->u
        if uv:
            current_state = 1
            alternatives = (0, 2)
        elif vu:
            current_state = 2
            alternatives = (0, 1)
        else:
            current_state = 0
            alternatives = (1, 2)
        proposed_state = alternatives[int(rng.integers(2))]
        # ---- propose "none" (edge deletion) --------------------------------
        if proposed_state == 0:
            new_adj = A.copy()
            new_adj[u, v] = 0
            new_adj[v, u] = 0
            return new_adj, 0.0, 0.0
        # ---- propose u->v ---------------------------------------------------
        if proposed_state == 1:
            if current_state == 2:
                # reversing v->u -> u->v: remove v->u then check if v reaches u
                would_cycle = _has_path(v, u, ignore_edge=(v, u))
            else:
                # adding u->v: cycle iff v can reach u
                would_cycle = _has_path(v, u, ignore_edge=None)
            if would_cycle:
                return A.copy(), 0.0, 0.0
            new_adj = A.copy()
            new_adj[u, v] = 1
            new_adj[v, u] = 0
            return new_adj, 0.0, 0.0
        # ---- propose v->u ---------------------------------------------------
        else:
            if current_state == 1:
                # reversing u->v -> v->u: remove u->v then check if u reaches v
                would_cycle = _has_path(u, v, ignore_edge=(u, v))
            else:
                # adding v->u: cycle iff u can reach v
                would_cycle = _has_path(u, v, ignore_edge=None)
            if would_cycle:
                return A.copy(), 0.0, 0.0
            new_adj = A.copy()
            new_adj[v, u] = 1
            new_adj[u, v] = 0
            return new_adj, 0.0, 0.0

    def propose_new_weights(self,
                            adjacency_matrix,
                            weight_matrix,
                            edges,
                            rng=None):
        if rng is None:
            rng = np.random.default_rng()
        new_adj = np.asarray(adjacency_matrix, dtype=int).copy()
        new_w   = np.asarray(weight_matrix, dtype=float).copy()
        if len(edges) == 0:
            return new_adj, new_w, 0, 0
        cand_vs = sorted({int(v) for (_, v) in edges})
        if len(cand_vs) == 0:
            return new_adj, new_w, 0, 0
        v = int(rng.choice(cand_vs))
        pa_new, b_new = self._sample_node_weight_block(new_adj, v, rng)
        new_w[:, v] = 0
        if len(pa_new) > 0:
            new_w[pa_new, v] = b_new
        new_w[new_adj == 0] = 0
        np.fill_diagonal(new_w, 0)
        return new_adj, new_w, 0.0, 0.0

    def propose_new_state_Structure_MCMC(self, adjacency_matrix, weight_matrix, p_structure=0.5, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() < p_structure:
            # ---- structure proposal (no weights here) ----
            new_adj, log_qG_fwd, log_qG_rev = self.propose_new_structure_only(adjacency_matrix, rng=rng)
            # ---- S1: refresh all nodes whose parent sets changed ----
            new_w, log_qw_fwd, log_qw_rev, _changed = self._refresh_weights_S1(
                np.asarray(adjacency_matrix, dtype=int),
                np.asarray(weight_matrix, dtype=float),
                np.asarray(new_adj, dtype=int),
                rng,
            )
            logp_prop = log_qG_fwd + log_qw_fwd
            logp_cur  = log_qG_rev + log_qw_rev
            return new_adj, new_w, logp_prop, logp_cur, "structure"
        else:
            edges = list(zip(*np.where(np.asarray(adjacency_matrix, dtype=int) == 1)))
            new_adj, new_w, p_prop, p_cur = self.propose_new_weights(
                adjacency_matrix, weight_matrix, edges, rng=rng
            )
            return new_adj, new_w, p_prop, p_cur, "weight"

    def propose_new_state_PARNI(self, LA, weight_matrix, ctx, rng=None, p_structure=0.5):
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() < p_structure:
            old_adj = LA.curr.astype(int)
            info = parni_step_one(LA, ctx, rng=rng, proposal_only=True)
            LA_prop     = info["LA_prop"]
            new_adj     = LA_prop.curr.astype(int)
            log_qG_fwd  = info["log_qG_fwd"]
            log_qG_rev  = info["log_qG_rev"]
            try:
                ctx["_last_parni_diag"] = {
                    "k_size": float(info.get("k_raw_size", info.get("k_size", 0.0))),
                    "R": float(info.get("k_total_groups", 0.0)),
                    "n_eval": float(info.get("n_eval", 0.0)),
                    "omega_thin_before": float(info.get("omega_thin_before", ctx.get("omega_thin", 0.0))),
                    "omega_thin_after": float(info.get("omega_thin_after", ctx.get("omega_thin", 0.0))),
                    "omega": float(info.get("omega_thin_after", ctx.get("omega_thin", 0.0))),
                }
            except Exception:
                pass
            # ---- S1: refresh all nodes whose parent sets changed ----
            new_w, log_qw_fwd, log_qw_rev, _changed = self._refresh_weights_S1(
                old_adj,
                np.asarray(weight_matrix, dtype=float),
                new_adj,
                rng,
            )
            logp_prop = log_qG_fwd + log_qw_fwd
            logp_cur  = log_qG_rev + log_qw_rev
            return new_adj, new_w, logp_prop, logp_cur, LA_prop, "structure"
        else:
            current_adj = LA.curr.astype(int)
            edges = list(zip(*np.where(current_adj == 1)))
            new_adj = current_adj
            new_w = np.asarray(weight_matrix, dtype=float).copy()
            LA_prop = LA
            if len(edges) == 0:
                return new_adj, new_w, 0.0, 0.0, LA_prop, "weight"
            new_adj, new_w, log_qk_fwd, log_qk_rev = self.propose_new_weights(
                new_adj, new_w, edges=edges, rng=rng
            )
            return new_adj, new_w, log_qk_fwd, log_qk_rev, LA_prop, "weight"

    def mcmc_sampling(self,
                    initial_adj,
                    initial_w,
                    iterations=5000,
                    burn_in=500,
                    level=0,
                    adapt_step=True,
                    rng=None):
        """
        单条链的 MCMC 抽样，返回：
        - current_adj: 最后一个样本的邻接矩阵
        - current_w:   最后一个样本的权重矩阵
        - acc_rate:    整体接受率
        - acc_structure: 仅结构 move 的接受率（非 joint kernel 时）
        - acc_weight:     仅权重 move 的接受率（非 joint kernel 时）
        """
        if rng is None:
            rng = np.random.default_rng()

        acceptance_ratios = []

        current_adj = np.array(initial_adj, copy=True)
        current_w   = np.array(initial_w, copy=True)

        # Initial splitting score
        current_ace = self._splitting_metric(current_w)

        num_edges = float(np.sum(current_adj))
        logit_p = np.log(self.p_edge / (1.0 - self.p_edge))
        log_prior_g = num_edges * logit_p
        if np.isneginf(log_prior_g) or np.isnan(log_prior_g):
            log_prior_g = -100.0
        mll_score = self.bge_model.mll(current_adj, self.data)
        current_log_graph = log_prior_g + mll_score
        p_G = log_prior_g
        p_X_G = mll_score

        # ========= NEW: branch-specific step sizes =========
        ACC = 0
        ACC_weight = 0
        ACC_structure = 0
        weight_moves = 0
        structure_moves = 0
        parni_ctx = None
        if self.structure_kernel == "PARNI":
            parni_ctx = self._clone_parni_ctx_for_chain(self.parni_ctx, burn_in=int(burn_in))
            LA = parni_make_LA_from_G(current_adj.astype(int), parni_ctx)
        # --- PARNI per-chain diagnostics (for outer-iter logging) ---
        parni_diag_k = []
        parni_diag_R = []
        parni_diag_n_eval = []
        parni_diag_omega = []
        # ------------------------------------------------------------
        # ========= NEW: 记录不同 branch 的接受率 =========
        acc_history_weight = []  # 非 joint 时，权重 move 的接受率
        acc_history_joint  = []  # joint kernel 时的接受率

        for it in range(iterations):
            if self.structure_kernel == "Structure_MCMC":
                new_adj, new_w, logp_prop, logp_cur, move = self.propose_new_state_Structure_MCMC(
                    current_adj, current_w, p_structure=self.p_structure, rng=rng
                )
                move_type = move 
                pairwise_effect = self._splitting_metric(new_w)
                if pairwise_effect < level:
                    acceptance_ratio = 0.0
                else:
                    if move_type == "weight":
                        acceptance_ratio = 1.0
                        proposed_ace = pairwise_effect
                        log_prior_g = p_G
                        mll_score = p_X_G
                        proposed_log_graph = current_log_graph
                    else:
                        # 结构 move：只用 log p(G)+log p(X|G) + 结构 proposal 比值
                        num_edges = float(np.sum(new_adj))
                        logit_p = np.log(self.p_edge / (1.0 - self.p_edge))
                        log_prior_g = num_edges * logit_p
                        if np.isneginf(log_prior_g) or np.isnan(log_prior_g):
                            log_prior_g = -100.0
                        mll_score = self.bge_model.mll(new_adj, self.data)
                        proposed_log_graph = log_prior_g + mll_score
                        log_acceptance_ratio = (proposed_log_graph + logp_cur) - (current_log_graph + logp_prop)
                        acceptance_ratio = 1.0 if log_acceptance_ratio >= 0 else float(np.exp(log_acceptance_ratio))
                        proposed_ace = pairwise_effect

            elif self.structure_kernel == "PARNI":
                LA_prev = LA
                new_adj, new_w, logp_prop, logp_cur, LA_prop,move = self.propose_new_state_PARNI(
                    LA, current_w, ctx=parni_ctx, rng=rng, p_structure=self.p_structure
                )
                # PARNI diag: capture k/R/omega/n_eval from the last structure proposal
                if move == "structure" and isinstance(parni_ctx, dict):
                    dlast = parni_ctx.get("_last_parni_diag", None)
                    if isinstance(dlast, dict):
                        if dlast.get("k_size") is not None:
                            parni_diag_k.append(float(dlast["k_size"]))
                        if dlast.get("R") is not None:
                            parni_diag_R.append(float(dlast["R"]))
                        if dlast.get("n_eval") is not None:
                            parni_diag_n_eval.append(float(dlast["n_eval"]))
                        # prefer omega_thin_after if present
                        if dlast.get("omega_thin_after") is not None:
                            parni_diag_omega.append(float(dlast["omega_thin_after"]))
                        elif dlast.get("omega") is not None:
                            parni_diag_omega.append(float(dlast["omega"]))
                move_type = move 
                LA_prop_llh = float(LA_prop.llh)
                pairwise_effect = self._splitting_metric(new_w)
                if pairwise_effect < level:
                    acceptance_ratio = 0.0
                    LA = LA_prev
                else:
                    if move_type == "weight":
                        acceptance_ratio = 1.0
                        proposed_ace = pairwise_effect
                        log_prior_g = p_G
                        mll_score = p_X_G
                        proposed_log_graph = current_log_graph
                    else:
                        num_edges = float(np.sum(new_adj))
                        logit_p = np.log(self.p_edge / (1.0 - self.p_edge))
                        log_prior_g = num_edges * logit_p
                        if np.isneginf(log_prior_g) or np.isnan(log_prior_g):
                            log_prior_g = -100.0
                        mll_score = LA_prop_llh
                        proposed_log_graph = log_prior_g + mll_score
                        log_acceptance_ratio = (proposed_log_graph + logp_cur) - (current_log_graph + logp_prop)
                        acceptance_ratio = 1.0 if log_acceptance_ratio >= 0 else float(np.exp(log_acceptance_ratio))
                        proposed_ace = pairwise_effect

            # ----------------- MH accept/reject -----------------
            if rng.random() < acceptance_ratio:
                current_adj = new_adj
                current_w   = new_w
                current_log_graph = proposed_log_graph
                current_ace = proposed_ace
                p_G = log_prior_g
                p_X_G = mll_score
                ACC += 1

                if self.structure_kernel == "PARNI":
                    LA = LA_prop
                if move == "structure":
                    ACC_structure += 1
                    structure_moves += 1
                elif move == "weight":
                    ACC_weight += 1
                    weight_moves += 1
            else:
                if move == "structure":
                    structure_moves += 1
                elif move == "weight":
                    weight_moves += 1
                if self.structure_kernel == "PARNI":
                    LA = LA_prev
            # ---- (D) paper §3.2 Eq(9): adaptive η / PIPs update (call AFTER MH decision) ----
            if self.structure_kernel == "PARNI" and move == "structure":
                diag_pips = parni_update_pips_eq9(parni_ctx, LA.curr)
            if move == "weight":
                acc_history_weight.append(acceptance_ratio)

        # ----------------- 汇总接受率 -----------------
        acc_rate = ACC / float(iterations)
        acc_structure_rate = ACC_structure / float(structure_moves) if structure_moves > 0 else 0.0
        acc_weight_rate    = ACC_weight    / float(weight_moves)    if weight_moves    > 0 else 0.0

        # --- aggregate PARNI diagnostics for this chain ---
        parni_diag = None
        if self.structure_kernel == "PARNI":
            try:
                parni_diag = {
                    "k_size_mean": float(np.mean(parni_diag_k)) if len(parni_diag_k) else 0.0,
                    "R_mean": float(np.mean(parni_diag_R)) if len(parni_diag_R) else 0.0,
                    "n_eval_mean": float(np.mean(parni_diag_n_eval)) if len(parni_diag_n_eval) else 0.0,
                    "omega_thin_mean": float(np.mean(parni_diag_omega)) if len(parni_diag_omega) else (float(parni_ctx.get("omega_thin", 0.0)) if isinstance(parni_ctx, dict) else 0.0),
                    "omega_thin_last": float(parni_ctx.get("omega_thin", 0.0)) if isinstance(parni_ctx, dict) else 0.0,
                    "num_struct_moves": int(len(parni_diag_k)),
                }
            except Exception:
                parni_diag = None

        return current_adj, current_w, acc_rate, acc_structure_rate, acc_weight_rate, parni_diag

    def compute_sample_graph_parallel(self, G, W, mcmc_iterations, level, seed=None):
        """
        Each parallel job constructs its own RNG and passes it down.
        MCMC stage will NOT use self.rng.
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        current_adj, current_w, acc_rate, acc_structure_rate, acc_weight_rate, parni_diag = self.mcmc_sampling(
            initial_adj=G,
            initial_w=W,
            iterations=mcmc_iterations,
            burn_in=int(mcmc_iterations * 0.1),
            level=level,
            rng=rng,
        )
        return current_adj, current_w, acc_rate, acc_structure_rate, acc_weight_rate, parni_diag

    def _resample_balanced(self, G_list, W_list, n):
        K = len(G_list)
        if K == 0:
            raise ValueError("No survivors to resample from (K=0).")
        rep = n // K
        rem = n % K
        idx = np.repeat(np.arange(K), rep)
        if rem > 0:
            extra = self.rng.choice(K, size=rem, replace=False)
            idx = np.concatenate([idx, extra])
        self.rng.shuffle(idx)
        G_new = [np.array(G_list[i], copy=True) for i in idx]
        W_new = [np.array(W_list[i], copy=True) for i in idx]
        return G_new, W_new, idx

    def _edge_mu_sigma(self, adj: np.ndarray, u: int, v: int):
        """
        Return (mu, sigma) for edge u->v under CURRENT parent set of v in adj.
        Uses node-block posterior params, and extracts the coordinate for u in Pa(v).
        Falls back to (self.mu, self.sigma) if ill-conditioned or u not in parents.
        """
        parents, loc, shape, df = self._node_weight_posterior_params(adj, v)
        if (parents is None) or (len(parents) == 0):
            return float(self.mu), float(self.sigma)
        if u not in parents:
            return float(self.mu), float(self.sigma)
        if loc is None or shape is None:
            return float(self.mu), float(self.sigma)
        idx = parents.index(u)
        mu = float(loc[idx])
        var = float(shape[idx, idx])
        if (not np.isfinite(mu)) or (not np.isfinite(var)) or var <= 0.0:
            return float(self.mu), float(self.sigma)
        return mu, float(np.sqrt(var))

    def _save_level_samples_arrays(
        self,
        iteration: int,
        level_value: float,
        adjs,
        Ws=None,
        ace=None,
        kind: str = "survivors",
    ) -> None:
        if not getattr(self, "save_level_samples", False):
            return
        out_dir = getattr(self, "level_samples_dir", None)
        if out_dir is None:
            out_dir = Path(self.save_dir) / "level_samples"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.level_samples_dir = out_dir
        it = int(iteration)
        tag = level_tag(level_value)
        adj_arr = np.asarray(adjs, dtype=np.uint8)
        np.save(out_dir / f"{kind}_adj_iter{it:03d}.npy", adj_arr)
        if getattr(self, "save_level_weights", False) and Ws is not None:
            W_arr = np.asarray(Ws, dtype=np.float32)
            np.save(out_dir / f"{kind}_W_iter{it:03d}.npy", W_arr)
        if ace is not None:
            ace_arr = np.asarray(ace, dtype=np.float32)
            np.save(out_dir / f"{kind}_ace_iter{it:03d}.npy", ace_arr)

    def calculate_probability(self, n, mcmc_iterations=5000):
        """
        Multilevel (or adaptive) filtering. This only shows the same overall logic as the original,
        with the core difference in compute_sample_graph_parallel.
        Here, we sample (adj, w), and can still use the i->j causal effect for layered filtering.
        """
        with open(self.output_file, "a") as f:
            log_and_print("Starting adaptive leveling...",f)
            log_and_print(f"(T={mcmc_iterations}, n param={n}) (kernel method: {self.structure_kernel})", f)
        probability_list = []
        S = 0  
        iteration = 0
        idx_target = 0
        current_level = -np.inf
        last_level = -np.inf
        num_target = len(self.X)
        suffix = f"(n={n})"
        level_trace = []
        trace_path = self.save_dir / f"level_trace_{suffix}.json"
        G_samples = sample_random_graphs(n, self.data.shape[1], p=self.p_edge)
        W_samples = list(self.initialize_edge_weight_matrix(G_samples))
        while True:
            if (self.max_outer_iter is not None) and (iteration >= self.max_outer_iter):
                remaining = len(self.X) - idx_target
                probability_list.extend([-50] * remaining)
                print("reach max_outer_iter, break")
                # dump level trace before returning
                try:
                    atomic_json_dump(level_trace, trace_path)
                except Exception:
                    pass
                return probability_list
            with open(self.output_file, "a") as f:
                log_and_print(f"Current level (metric)={current_level} | {self._condition_str(current_level)}", f)
            iteration += 1
            results = []
            seeds = self.rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32)

            with tqdm_joblib(tqdm(desc="Parallel ACE Computation", total=n)) as progress_bar:
                results = Parallel(n_jobs=-1)(
                    delayed(self.compute_sample_graph_parallel)(G, W, mcmc_iterations, current_level, int(sd))
                    for (G, W, sd) in zip(G_samples, W_samples, seeds)
                )
            index = 0
            tmp = 0
            mark = True
            rho = 0.1
            k = int(np.ceil(rho * n))
            k = min(max(k, 1), n)  
            new_adjs = [r[0] for r in results]
            new_ws = [r[1] for r in results]
            acc_rates = [r[2] for r in results]
            acc_structure_rates = [r[3] for r in results]
            acc_weight_rates = [r[4] for r in results]
            parni_chain_diags = [r[5] for r in results]
            parni_chain_diags = [r[5] for r in results]  # may be None for non-PARNI kernels
            with open(self.output_file, "a") as f:
                log_and_print(f"Average acceptance rate in this iteration: {np.mean(acc_rates)}", f)
                log_and_print(f"Average structure acceptance rate in this iteration: {np.mean(acc_structure_rates)}", f)
                log_and_print(f"Average weight acceptance rate in this iteration: {np.mean(acc_weight_rates)}", f)
                if self.structure_kernel == "PARNI":
                    try:
                        ks = np.array([d.get("k_size_mean", 0.0) for d in parni_chain_diags if isinstance(d, dict)], dtype=float)
                        Rs = np.array([d.get("R_mean", 0.0) for d in parni_chain_diags if isinstance(d, dict)], dtype=float)
                        nes = np.array([d.get("n_eval_mean", 0.0) for d in parni_chain_diags if isinstance(d, dict)], dtype=float)
                        oms = np.array([d.get("omega_thin_last", 0.0) for d in parni_chain_diags if isinstance(d, dict)], dtype=float)
                        if ks.size:
                            log_and_print(f"[PARNI-Diag] k_size(mean over chain) mean={float(np.mean(ks)):.3g}, q50={float(np.median(ks)):.3g}, q90={float(np.percentile(ks,90)):.3g}", f)
                        if Rs.size:
                            log_and_print(f"[PARNI-Diag] R(groups) mean={float(np.mean(Rs)):.3g}, q50={float(np.median(Rs)):.3g}, q90={float(np.percentile(Rs,90)):.3g}", f)
                        if oms.size and nes.size:
                            log_and_print(f"[PARNI-Diag] omega_thin(last) mean={float(np.mean(oms)):.3g}; n_eval(mean over chain) mean={float(np.mean(nes)):.3g}", f)
                    except Exception as e:
                        log_and_print(f"[PARNI-Diag] failed to aggregate: {repr(e)}", f)
            Graph_pair_with_ace = []
            metric_details = [self._splitting_metric_details(w) for w in new_ws]
            score_list = [md[0] for md in metric_details]
            metric_vec_list = [md[2] for md in metric_details]
            for adj, w, sc, mvec in zip(new_adjs, new_ws, score_list, metric_vec_list):
                Graph_pair_with_ace.append([sc, adj, w, mvec])
            ace_list = [x[0] for x in Graph_pair_with_ace]
            sorted_graph_pair_with_ace = sorted(Graph_pair_with_ace, key=lambda x: x[0], reverse=True)
            next_level = sorted_graph_pair_with_ace[k - 1][0]   # kth-largest (top rho fraction threshold)

            with open(self.output_file, "a") as f:
                log_and_print(
                    f"Next level selected at rho={rho}: level(metric)={next_level}, k={k} | {self._condition_str(next_level)}",
                    f
                )
            last_level = current_level
            current_level = next_level
            #-----------------------
            bins = 30
            plt.figure(figsize=(10, 6))
            plt.hist(ace_list, bins=bins, color='blue', edgecolor='black', alpha=0.7)
            plt.axvline(x=current_level, color='red', linestyle='--', linewidth=2, label=f'Current_level: {current_level}')
            for target_value in self.X[idx_target:]:
                plt.axvline(x=target_value, color='green', linestyle='--', linewidth=2, label=f'Target ACE: {target_value}')
            plt.axvline(x=last_level, color='yellow', linestyle='--', linewidth=2, label=f'Last_level: {last_level}')
            plt.title('Distribution of ACE Values (Adaptive)')
            plt.xlabel('ACE Value')
            plt.ylabel('Frequency')
            plt.legend()
            filename = f'{self.save_dir}/adaptive_ace_values_{suffix}(level={current_level}).png'
            plt.savefig(filename)
            plt.close()
            #-----------------------
            proportion = sum(1 for item in sorted_graph_pair_with_ace if item[0] >= current_level) / len(sorted_graph_pair_with_ace)
            if proportion == 0:
                try:
                    trace_entry["idx_target_end"] = int(idx_target)
                    level_trace.append(trace_entry)
                    atomic_json_dump(level_trace, trace_path)
                except Exception:
                    pass
                return 0
            surv_items = [item for item in sorted_graph_pair_with_ace if item[0] >= current_level]
            new_G_samples = [item[1] for item in surv_items]
            new_W_samples = [item[2] for item in surv_items]
            ace_surv_sorted = [item[0] for item in surv_items]
            if getattr(self, "save_level_samples", False):
                try:
                    self._save_level_samples_arrays(
                        iteration=iteration,
                        level_value=current_level,
                        adjs=new_G_samples,
                        Ws=new_W_samples,
                        ace=ace_surv_sorted,
                        kind="survivors",
                    )
                    if getattr(self, "save_level_all", False):
                        self._save_level_samples_arrays(
                            iteration=iteration,
                            level_value=current_level,
                            adjs=new_adjs,
                            Ws=new_ws,
                            ace=ce_list,
                            kind="all",
                        )
                except Exception as e:
                    with open(self.output_file, "a") as f:
                        log_and_print(f"[LevelSamples] save failed: {repr(e)}", f)            
            # ---------------- Level trace (per-iteration diagnostics) ----------------
            ace_arr = np.asarray(ace_list, dtype=float)
            surv_mask = ace_arr >= current_level
            ace_surv = ace_arr[surv_mask]

            # graph edge count summary
            edge_counts_all = np.array([int(np.sum(np.asarray(a, dtype=int))) for a in new_adjs], dtype=float)
            edge_counts_surv = np.array([int(np.sum(np.asarray(a, dtype=int))) for a in new_G_samples], dtype=float)

            trace_entry = {
                "iteration": int(iteration),
                "n": int(n),
                "rho": float(rho),
                "k": int(k),
                "idx_target_start": int(idx_target),
                "last_level": float(last_level) if np.isfinite(last_level) else None,
                "current_level": float(current_level) if np.isfinite(current_level) else None,
                "proportion": float(proportion),
                "S_before": float(S),
                "S_after": float(S + np.log(proportion)) if proportion > 0 else None,
                "acyclic_all": bool(mark),
                "acc_rate_mean": float(np.mean(acc_rates)) if len(acc_rates) else None,
                "acc_rate_std": float(np.std(acc_rates)) if len(acc_rates) else None,
                "acc_structure_mean": float(np.mean(acc_structure_rates)) if len(acc_structure_rates) else None,
                "acc_structure_std": float(np.std(acc_structure_rates)) if len(acc_structure_rates) else None,
                "acc_weight_mean": float(np.mean(acc_weight_rates)) if len(acc_weight_rates) else None,
                "acc_weight_std": float(np.std(acc_weight_rates)) if len(acc_weight_rates) else None,
                "ace_all": summarize_array(ace_arr),
                "ace_survivors": summarize_array(ace_surv),
                "edge_counts_all": summarize_array(edge_counts_all),
                "edge_counts_survivors": summarize_array(edge_counts_surv),
                "unique_graphs_all": int(unique_graph_count(new_adjs)),
                "unique_graphs_survivors": int(unique_graph_count(new_G_samples)),
                "num_survivors": int(len(new_G_samples)),
                "targets_resolved": [],
            }

            # ---------------- SanityCheck: Level-end survivor weight diagnostics ----------------
            if getattr(self, "enable_extra_sanity", False):
                try:
                    # survivors are exactly new_G_samples/new_W_samples at this point
                    surv_W = new_W_samples
                    surv_G = new_G_samples
                    m = len(surv_W)
                    if m > 0:
                        max_abs_list = []
                        # track target edge stats if present
                        target_u, target_v = int(self.i), int(self.j)
                        tgt_present = 0
                        tgt_w_vals = []
                        tgt_z_vals = []

                        for Gs, Ws in zip(surv_G, surv_W):
                            Ws = np.asarray(Ws, dtype=float)
                            max_abs_list.append(float(np.max(np.abs(Ws))))

                            # target edge
                            if int(Gs[target_u, target_v]) == 1:
                                tgt_present += 1
                                w_uv = float(Ws[target_u, target_v])
                                mu_uv, sigma_uv = self._edge_mu_sigma(Gs, target_u, target_v)
                                z = (w_uv - mu_uv) / sigma_uv if sigma_uv > 0 else np.nan
                                tgt_w_vals.append(w_uv)
                                tgt_z_vals.append(z)

                        max_abs_arr = np.array(max_abs_list, dtype=float)
                        q10, q50, q90 = np.percentile(max_abs_arr, [10, 50, 90])

                        with open(self.output_file, "a") as f:
                            log_and_print(
                                f"[W1S1-Sanity-LvlEnd] iter={iteration} level={current_level:.6g} "
                                f"survivors={m}/{n} max|W|: q10={q10:.6g} median={q50:.6g} q90={q90:.6g} "
                                f"max={float(max_abs_arr.max()):.6g}",
                                f
                            )

                            # target edge summary
                            if tgt_present > 0:
                                w_arr = np.array(tgt_w_vals, dtype=float)
                                z_arr = np.array(tgt_z_vals, dtype=float)
                                log_and_print(
                                    f"[W1S1-Sanity-LvlEnd] edge({target_u}->{target_v}) present {tgt_present}/{m}: "
                                    f"w mean={float(np.mean(w_arr)):.6g}, std={float(np.std(w_arr)):.6g}, "
                                    f"min={float(np.min(w_arr)):.6g}, max={float(np.max(w_arr)):.6g}; "
                                    f"z mean={float(np.nanmean(z_arr)):.3f}, min={float(np.nanmin(z_arr)):.3f}, max={float(np.nanmax(z_arr)):.3f}",
                                    f
                                )
                            else:
                                log_and_print(
                                    f"[W1S1-Sanity-LvlEnd] edge({target_u}->{target_v}) present 0/{m} among survivors.",
                                    f
                                )

                            # print top-|w| edges for ONE representative survivor (the top-ACE one)
                            top_item = sorted_graph_pair_with_ace[0]
                            top_adj, top_w = top_item[1], top_item[2]
                            top_w = np.asarray(top_w, dtype=float)
                            d = top_w.shape[0]

                            # list all directed edges present
                            edges_present = list(zip(*np.where(np.asarray(top_adj, dtype=int) == 1)))
                            if edges_present:
                                # compute abs weights and pick top-k
                                abs_w = np.array([abs(top_w[u, v]) for (u, v) in edges_present], dtype=float)
                                k = min(int(getattr(self, "sanity_topk_edges", 8)), len(edges_present))
                                idxs = np.argsort(-abs_w)[:k]

                                log_and_print(f"[W1S1-Sanity-LvlEnd] top-{k} |w| edges in TOP-ACE survivor:", f)
                                for t in idxs:
                                    u, v = edges_present[int(t)]
                                    w_uv = float(top_w[u, v])
                                    mu_uv, sigma_uv = self._edge_mu_sigma(top_adj, int(u), int(v))
                                    z = (w_uv - mu_uv) / sigma_uv if sigma_uv > 0 else np.nan
                                    log_and_print(
                                        f"  edge({int(u)}->{int(v)}): w={w_uv:.6g}, mu={mu_uv:.6g}, sigma={sigma_uv:.6g}, z={z:.3f}",
                                        f
                                    )

                except Exception as e:
                    with open(self.output_file, "a") as f:
                        log_and_print(f"[W1S1-Sanity-LvlEnd] failed: {repr(e)}", f)
            # ----------------------------------------------------------------------
            with open(self.output_file, "a") as f:
                log_and_print(f"90th Percentile {current_level} proportion: {proportion}", f)
            #-----------------------
            last_valid_idx = None
            for i, target_value in enumerate(self.X[idx_target:], start=idx_target):
                if current_level >= target_value:
                    last_valid_idx = i
                    count_exceed = sum(1 for item in sorted_graph_pair_with_ace if item[0] > target_value)
                    final_proportion = count_exceed / n
                    with open(self.output_file, "a") as f:
                        log_and_print(f"Level {current_level} reaches target {self._condition_str(target_value)}; final proportion={final_proportion}", f)
                    if final_proportion == 0:
                        tmp = -50.0
                    else:
                        tmp = S + np.log(final_proportion)
                    probability_list.append(tmp)
                    try:
                        trace_entry["targets_resolved"].append({
                            "target_value": float(target_value),
                            "count_exceed": int(count_exceed),
                            "final_proportion": float(final_proportion),
                            "logp": float(tmp),
                        })
                    except Exception:
                        pass
                    sel_items = [item for item in sorted_graph_pair_with_ace if item[0] > target_value]
                    graphs = [item[1] for item in sel_items]
                    weights = [item[2] for item in sel_items]
                    # For plotting we reuse the first constraint's metric values (sign * CE).
                    ce_samples = [float(item[3][0]) for item in sel_items]
                    if len(ce_samples) > 0:
                        ce_samples = np.asarray(ce_samples, dtype=float)
                        gs_np = np.stack(graphs)
                        thetas_np = np.stack(weights)
                        cond_label = self._condition_str(target_value, with_indices=True)
                        thr0 = float(self._ce_thr_metric[0] + float(target_value) * float(self._ce_scales[0]))
                        save_path = self.save_dir / f"multiCE_score_ge_{float(target_value):.4f}"
                        save_path.mkdir(parents=True, exist_ok=True)
                        plot_edge_frequency_and_weight_avg_graphs(
                            ce_samples=ce_samples,
                            gs_np=gs_np,
                            thetas_np=thetas_np,
                            i=self.i,
                            j=self.j,
                            threshold=thr0,
                            save_path=save_path,
                            file_name=self.output_file,
                            condition_label=cond_label
                        )
            if last_valid_idx is not None:
                idx_target = last_valid_idx+1 
                if idx_target == len(self.X):
                    np.save(self.save_dir / f"samples_graphs_level_{current_level}.npy", new_G_samples)
                    np.save(self.save_dir / f"samples_weights_level_{current_level}.npy", new_W_samples)
                    try:
                        atomic_json_dump(level_trace, trace_path)
                    except Exception:
                        pass
                    return probability_list
            #-----------------------
            try:
                trace_entry["idx_target_end"] = int(idx_target)
                level_trace.append(trace_entry)
                atomic_json_dump(level_trace, trace_path)
            except Exception:
                pass
            S += np.log(proportion)
            G_samples, W_samples, _ = self._resample_balanced(new_G_samples, new_W_samples, n)

        return np.exp(S)

if __name__ == "__main__":
    dataset = "sachs"      # "sachs" or "synthetic"
    mean = 0
    X_levels = [0.0]
    n = 500
    mcmc_iterations = 5000
    for d in [11]:        
        if d==4:
            eps1 = 0.5
            eps2 = 0.2
            ce_constraints_list = [
                ("weak", [
                    (0, 1, ">", -2.69215861 - eps1, 1.0),
                    (0, 1, "<", -2.69215861 + eps1, 1.0),
                    (3, 2, ">",  3.06410405 - eps1, 1.0),
                    (3, 2, "<",  3.06410405 + eps1, 1.0),
                ]),
                ("strong", [
                    (2, 1, ">",  1.13723718 - eps2, 1.0), (2, 1, "<",  1.13723718 + eps2, 1.0),
                    (3, 1, ">",  2.07523587 - eps2, 1.0), (3, 1, "<",  2.07523587 + eps2, 1.0),
                    (3, 2, ">",  3.06410405 - eps2, 1.0), (3, 2, "<",  3.06410405 + eps2, 1.0),
                    (3, 0, ">", -0.86398111 - eps2, 1.0), (3, 0, "<", -0.86398111 + eps2, 1.0),
                    (0, 1, ">", -2.69215861 - eps2, 1.0), (0, 1, "<", -2.69215861 + eps2, 1.0),
                ])
            ]
        
        elif d==8:
            eps = 0.3
            ce_constraints_list = [
                ("weak", [
                        (1, 0, ">", -1.95311464179 - eps, 1.0),
                        (1, 0, "<", -1.95311464179 + eps, 1.0),

                        (2, 4, ">",  2.49083503925 - eps, 1.0),
                        (2, 4, "<",  2.49083503925 + eps, 1.0),

                        (5, 6, ">",  4.24338751225 - eps, 1.0),
                        (5, 6, "<",  4.24338751225 + eps, 1.0),

                        (7, 3, ">",  7.05561950734 - eps, 1.0),
                        (7, 3, "<",  7.05561950734 + eps, 1.0),
                ]),
                ("strong", [
                    # (7 -> 0) in [a-eps, a+eps]
                    (7, 0, ">", -0.3048072149513295 - eps, 1.0),
                    (7, 0, "<", -0.3048072149513295 + eps, 1.0),

                    # (1 -> 3)
                    (1, 3, ">", 16.027190054612383 - eps, 1.0),
                    (1, 3, "<", 16.027190054612383 + eps, 1.0),

                    # (2 -> 4)
                    (2, 4, ">", 2.490835039251952 - eps, 1.0),
                    (2, 4, "<", 2.490835039251952 + eps, 1.0),

                    # (5 -> 6)
                    (5, 6, ">", 4.243387512245249 - eps, 1.0),
                    (5, 6, "<", 4.243387512245249 + eps, 1.0),

                    # (1 -> 0)
                    (1, 0, ">", -1.9531146417858813 - eps, 1.0),
                    (1, 0, "<", -1.9531146417858813 + eps, 1.0),

                    # (2 -> 3)
                    (2, 3, ">", -3.3479795383329027 - eps, 1.0),
                    (2, 3, "<", -3.3479795383329027 + eps, 1.0),

                    # (5 -> 4)
                    (5, 4, ">", -1.1973893003588836 - eps, 1.0),
                    (5, 4, "<", -1.1973893003588836 + eps, 1.0),

                    # (7 -> 6)
                    (7, 6, ">", -2.1976200499204093 - eps, 1.0),
                    (7, 6, "<", -2.1976200499204093 + eps, 1.0),    
                    
                    (5, 0, ">", 0.5885526616002472 - eps, 1.0),
                    (5, 0, "<", 0.5885526616002472 + eps, 1.0),

                    # (1 -> 7)
                    (1, 7, ">", 2.273046083876654 - eps, 1.0),
                    (1, 7, "<", 2.273046083876654 + eps, 1.0),

                    # (2 -> 6)
                    (2, 6, ">", 2.7169531827413755 - eps, 1.0),
                    (2, 6, "<", 2.7169531827413755 + eps, 1.0),

                    # (4 -> 3)
                    (4, 3, ">", 1.266077983643409 - eps, 1.0),
                    (4, 3, "<", 1.266077983643409 + eps, 1.0),
                ])
            ]

        elif d==11:
            ce_constraints_list = [
                ("sachs",[
                        (6, 5, ">", 0.74, 1.0),
                        (1, 10, ">", 0.65, 1.0),
                ]),
            ]
        elif d==16:
            eps = 0.3
            ce_constraints_list = [
                ("weak", [
                    (0, 10, ">", -1.82416233657 - eps, 1.0),
                    (0, 10, "<", -1.82416233657 + eps, 1.0),

                    (2, 7, ">", -0.842063606707 - eps, 1.0),
                    (2, 7, "<", -0.842063606707 + eps, 1.0),

                    (3, 6, ">",  0.0347085224751 - eps, 1.0),
                    (3, 6, "<",  0.0347085224751 + eps, 1.0),

                    (8, 11, ">", -0.196858356185 - eps, 1.0),
                    (8, 11, "<", -0.196858356185 + eps, 1.0),

                    (9, 13, ">", -3.84971206075 - eps, 1.0),
                    (9, 13, "<", -3.84971206075 + eps, 1.0),

                    (12, 4, ">",  1.96423560056 - eps, 1.0),
                    (12, 4, "<",  1.96423560056 + eps, 1.0),

                    (14, 1, ">", -1.18299716545 - eps, 1.0),
                    (14, 1, "<", -1.18299716545 + eps, 1.0),

                    (15, 5, ">",  0.497990441127 - eps, 1.0),
                    (15, 5, "<",  0.497990441127 + eps, 1.0),
                ]),
                ("strong",[
                    # --- weak matching ---
                    (0, 10, ">", -1.82416233657 - eps, 1.0),
                    (0, 10, "<", -1.82416233657 + eps, 1.0),

                    (2, 7, ">", -0.842063606707 - eps, 1.0),
                    (2, 7, "<", -0.842063606707 + eps, 1.0),

                    (3, 6, ">",  0.0347085224751 - eps, 1.0),
                    (3, 6, "<",  0.0347085224751 + eps, 1.0),

                    (8, 11, ">", -0.196858356185 - eps, 1.0),
                    (8, 11, "<", -0.196858356185 + eps, 1.0),

                    (9, 13, ">", -3.84971206075 - eps, 1.0),
                    (9, 13, "<", -3.84971206075 + eps, 1.0),

                    (12, 4, ">",  1.96423560056 - eps, 1.0),
                    (12, 4, "<",  1.96423560056 + eps, 1.0),

                    (14, 1, ">", -1.18299716545 - eps, 1.0),
                    (14, 1, "<", -1.18299716545 + eps, 1.0),

                    (15, 5, ">",  0.497990441127 - eps, 1.0),
                    (15, 5, "<",  0.497990441127 + eps, 1.0),

                    # --- extra matching (covers all nodes once again) ---
                    (0, 13, ">", -3.06765113805 - eps, 1.0),
                    (0, 13, "<", -3.06765113805 + eps, 1.0),

                    (2, 10, ">",  0.527005312514 - eps, 1.0),
                    (2, 10, "<",  0.527005312514 + eps, 1.0),

                    (5, 14, ">",  0.287955635252 - eps, 1.0),
                    (5, 14, "<",  0.287955635252 + eps, 1.0),

                    (8, 7, ">", -0.268777469806 - eps, 1.0),
                    (8, 7, "<", -0.268777469806 + eps, 1.0),

                    (9, 6, ">",  2.096867949 - eps, 1.0),
                    (9, 6, "<",  2.096867949 + eps, 1.0),

                    (11, 4, ">", -5.90423355655 - eps, 1.0),
                    (11, 4, "<", -5.90423355655 + eps, 1.0),

                    (12, 3, ">",  0.00644876590788 - eps, 1.0),
                    (12, 3, "<",  0.00644876590788 + eps, 1.0),

                    (15, 1, ">", -2.64437302486 - eps, 1.0),
                    (15, 1, "<", -2.64437302486 + eps, 1.0),
                ])
            ]
        
        elif d==32:
            eps = 0.3
            ce_constraints_list = [
                ("weak",  [
                    (0, 29, ">", -1.34505431535 - eps, 1.0),
                    (0, 29, "<", -1.34505431535 + eps, 1.0),

                    (3, 12, ">",  0.929933662401 - eps, 1.0),
                    (3, 12, "<",  0.929933662401 + eps, 1.0),

                    (4, 2, ">",  0.14240541553 - eps, 1.0),
                    (4, 2, "<",  0.14240541553 + eps, 1.0),

                    (5, 1, ">",  0.709386070145 - eps, 1.0),
                    (5, 1, "<",  0.709386070145 + eps, 1.0),

                    (6, 26, ">", -1.11779523106 - eps, 1.0),
                    (6, 26, "<", -1.11779523106 + eps, 1.0),

                    (8, 19, ">", -0.110506416384 - eps, 1.0),
                    (8, 19, "<", -0.110506416384 + eps, 1.0),

                    (9, 10, ">",  1.24907817749 - eps, 1.0),
                    (9, 10, "<",  1.24907817749 + eps, 1.0),

                    (17, 7, ">",  1.96673298782 - eps, 1.0),
                    (17, 7, "<",  1.96673298782 + eps, 1.0),

                    (21, 20, ">", -1.10883048447 - eps, 1.0),
                    (21, 20, "<", -1.10883048447 + eps, 1.0),

                    (22, 18, ">",  0.752394982232 - eps, 1.0),
                    (22, 18, "<",  0.752394982232 + eps, 1.0),

                    (23, 14, ">", -0.658357249436 - eps, 1.0),
                    (23, 14, "<", -0.658357249436 + eps, 1.0),

                    (24, 16, ">", -4.45911265557 - eps, 1.0),
                    (24, 16, "<", -4.45911265557 + eps, 1.0),

                    (25, 15, ">", -1.01445069984 - eps, 1.0),
                    (25, 15, "<", -1.01445069984 + eps, 1.0),

                    (27, 30, ">", -1.90987340795 - eps, 1.0),
                    (27, 30, "<", -1.90987340795 + eps, 1.0),

                    (28, 13, ">", -1.81031428023 - eps, 1.0),
                    (28, 13, "<", -1.81031428023 + eps, 1.0),

                    (31, 11, ">", -2.34803252702 - eps, 1.0),
                    (31, 11, "<", -2.34803252702 + eps, 1.0),
                ]),
                ("strong",[
                    # --- weak matching ---
                    (0, 29, ">", -1.34505431535 - eps, 1.0),
                    (0, 29, "<", -1.34505431535 + eps, 1.0),

                    (3, 12, ">",  0.929933662401 - eps, 1.0),
                    (3, 12, "<",  0.929933662401 + eps, 1.0),

                    (4, 2, ">",  0.14240541553 - eps, 1.0),
                    (4, 2, "<",  0.14240541553 + eps, 1.0),

                    (5, 1, ">",  0.709386070145 - eps, 1.0),
                    (5, 1, "<",  0.709386070145 + eps, 1.0),

                    (6, 26, ">", -1.11779523106 - eps, 1.0),
                    (6, 26, "<", -1.11779523106 + eps, 1.0),

                    (8, 19, ">", -0.110506416384 - eps, 1.0),
                    (8, 19, "<", -0.110506416384 + eps, 1.0),

                    (9, 10, ">",  1.24907817749 - eps, 1.0),
                    (9, 10, "<",  1.24907817749 + eps, 1.0),

                    (17, 7, ">",  1.96673298782 - eps, 1.0),
                    (17, 7, "<",  1.96673298782 + eps, 1.0),

                    (21, 20, ">", -1.10883048447 - eps, 1.0),
                    (21, 20, "<", -1.10883048447 + eps, 1.0),

                    (22, 18, ">",  0.752394982232 - eps, 1.0),
                    (22, 18, "<",  0.752394982232 + eps, 1.0),

                    (23, 14, ">", -0.658357249436 - eps, 1.0),
                    (23, 14, "<", -0.658357249436 + eps, 1.0),

                    (24, 16, ">", -4.45911265557 - eps, 1.0),
                    (24, 16, "<", -4.45911265557 + eps, 1.0),

                    (25, 15, ">", -1.01445069984 - eps, 1.0),
                    (25, 15, "<", -1.01445069984 + eps, 1.0),

                    (27, 30, ">", -1.90987340795 - eps, 1.0),
                    (27, 30, "<", -1.90987340795 + eps, 1.0),

                    (28, 13, ">", -1.81031428023 - eps, 1.0),
                    (28, 13, "<", -1.81031428023 + eps, 1.0),

                    (31, 11, ">", -2.34803252702 - eps, 1.0),
                    (31, 11, "<", -2.34803252702 + eps, 1.0),

                    # --- extra matching (covers all nodes once again) ---
                    (0, 22, ">", -2.42457150333 - eps, 1.0),
                    (0, 22, "<", -2.42457150333 + eps, 1.0),

                    (2, 29, ">",  0.311442950464 - eps, 1.0),
                    (2, 29, "<",  0.311442950464 + eps, 1.0),

                    (3, 26, ">", -0.176541068979 - eps, 1.0),
                    (3, 26, "<", -0.176541068979 + eps, 1.0),

                    (4, 28, ">", -0.324001992541 - eps, 1.0),
                    (4, 28, "<", -0.324001992541 + eps, 1.0),

                    (5, 11, ">",  0.409586131852 - eps, 1.0),
                    (5, 11, "<",  0.409586131852 + eps, 1.0),

                    (6, 1, ">", -0.667574074729 - eps, 1.0),
                    (6, 1, "<", -0.667574074729 + eps, 1.0),

                    (8, 25, ">",  0.0 - eps, 1.0),
                    (8, 25, "<",  0.0 + eps, 1.0),

                    (9, 13, ">", -0.37036049651 - eps, 1.0),
                    (9, 13, "<", -0.37036049651 + eps, 1.0),

                    (12, 14, ">",  0.430077297733 - eps, 1.0),
                    (12, 14, "<",  0.430077297733 + eps, 1.0),

                    (17, 15, ">",  1.45344400444 - eps, 1.0),
                    (17, 15, "<",  1.45344400444 + eps, 1.0),

                    (21, 10, ">", -0.456821060749 - eps, 1.0),
                    (21, 10, "<", -0.456821060749 + eps, 1.0),

                    (23, 20, ">", -0.00782826081176 - eps, 1.0),
                    (23, 20, "<", -0.00782826081176 + eps, 1.0),

                    (24, 19, ">", -0.952909851873 - eps, 1.0),
                    (24, 19, "<", -0.952909851873 + eps, 1.0),

                    (27, 18, ">",  7.62768061424 - eps, 1.0),
                    (27, 18, "<",  7.62768061424 + eps, 1.0),

                    (30, 16, ">", -1.55716470184 - eps, 1.0),
                    (30, 16, "<", -1.55716470184 + eps, 1.0),

                    (31, 7, ">",  2.97750676624 - eps, 1.0),
                    (31, 7, "<",  2.97750676624 + eps, 1.0),
                ])
            ]
        for name, ce_constraints in ce_constraints_list:
            for structure_kernel in ["PARNI", "Structure_MCMC"]:
                results = []
                for num in [1]:
                    for run_num in range(1,11):
                        seed = random.randint(0, 10000)
                        rng = np.random.default_rng(seed)
                        bge_model = BGe(d=d, alpha_u=10)
                        if dataset == "synthetic":
                            sub_dir = f"{structure_kernel}/d={d}(n=500,mcmc=5000)(2)/"
                            base_dir = (
                                f"PARNI-Structure_MCMC(mean=0)(multi-CE_{name})/" + sub_dir
                                if mean == 0
                                else f"PARNI-Structure_MCMC(Gamma)(multi-CE_{name})/" + sub_dir
                            )
                            save_dir = base_dir + f"case{num}/run_{run_num}"

                            load_dir = f"data/mean=0/d={d}/" if mean == 0 else f"data/mean=2/d={d}/"
                            load_case_dir = load_dir + f"case{num}"  # adjust if needed

                            G = np.load(f"{load_case_dir}/G_{d}Nodes_train_size_1000.npy")
                            X_train = np.load(f"{load_case_dir}/train_{d}Nodes_train_size_1000.npy")

                            # Optional: CE sanity (for each constraint)
                            ce = pairwise_linear_ce_no_params(
                                np.copy([G]), X_train, bge_model, params_per_graph=500, avg=True, return_B=False
                            )

                        elif dataset == "sachs":
                            sub_dir = f"{structure_kernel}/sachs_multiCE/"
                            base_dir = "PARNI-Structure_MCMC(sachs)/" + sub_dir
                            save_dir = base_dir + f"/run_{run_num}"

                            load_dir = "data/sachs/"
                            G = np.load(f"{load_dir}/sachs_graph.npy")
                            X_train = np.load(f"{load_dir}/sachs_data.npy")

                            # Optional: CE sanity (matrix)
                            B_samples, ce = pairwise_linear_ce_no_params(
                                np.copy([G]), X_train, bge_model, params_per_graph=5000, avg=True, return_B=True
                            )

                        else:
                            raise ValueError(f"Unknown dataset={dataset!r}")

                        print("seed:", seed)
                        print("X_train shape:", X_train.shape)
                        print(f"num of edges and num of nodes: {np.sum(G)}, {d}")

                        # Print approx CE for each constraint (diagnostic)
                        for (ii, jj, op, thr, sc) in ce_constraints:
                            print(f"approx CE({ii}->{jj}) = {ce[int(ii), int(jj)]:.6g}  constraint: CE {op} {thr}")

                        tmp = Path(save_dir)
                        tmp.mkdir(parents=True, exist_ok=True)
                        output_file = os.path.join(tmp, f"{structure_kernel}_run{run_num}_output_results.txt")

                        multilevel_model = Multilevel(
                            bge_model=bge_model,
                            data=X_train,
                            X=X_levels,
                            ce_constraints=ce_constraints,
                            save_dir=tmp,
                            output_file=output_file,
                            max_outer_iter=10,
                            rng=rng,
                            structure_kernel=structure_kernel,
                            p_structure=p_structure_schedule(d, T=mcmc_iterations),
                        )

                        probability_list = multilevel_model.calculate_probability(n, mcmc_iterations)

                        # ------------- Logging -------------
                        with open(output_file, "a") as f:
                            log_and_print(f"run {run_num}", f)
                            log_and_print(f"seed: {seed}", f)
                            log_and_print(f"constraints: {ce_constraints}", f)
                            for (ii, jj, op, thr, sc) in ce_constraints:
                                log_and_print(f"approx CE({ii}->{jj}) = {ce[int(ii), int(jj)]}", f)

                            for L, logp in zip(X_levels, probability_list):
                                log_and_print(f"------------------", f)
                                log_and_print(f"Score level L: {float(L)}", f)
                                log_and_print(f"Condition: {multilevel_model._condition_str(L)}", f)
                                log_and_print(f"P(score >= {float(L)}) = {np.exp(logp)}, e^{logp}", f)

                        results.append({
                            "run": run_num,
                            "seed": seed,
                            "X_levels": [float(x) for x in X_levels],
                            "log_probs": [float(x) for x in probability_list],
                            "probs": [float(np.exp(x)) for x in probability_list],
                        })

                        # ------------- Plot -------------
                        plt.figure(figsize=(8, 6))
                        plt.plot(X_levels, probability_list, marker="o", linestyle="-", label="log P(score >= L)")
                        texts = []
                        for x, y in zip(X_levels, probability_list):
                            texts.append(plt.text(x, y, f"{y:.2e}", fontsize=6))
                        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
                        plt.xlabel("Score level L")
                        plt.ylabel("log Probability")
                        plt.title("log P(score >= L) vs. L")
                        plt.legend()
                        plt.grid()
                        plt.savefig(str(tmp / "probability_vs_level(log).png"), dpi=300, bbox_inches="tight")
                        plt.close()

                        prob_linear = [np.exp(x) for x in probability_list]
                        plt.figure(figsize=(8, 6))
                        plt.plot(X_levels, prob_linear, marker="o", linestyle="-", label="P(score >= L)")
                        texts = []
                        for x, y in zip(X_levels, prob_linear):
                            texts.append(plt.text(x, y, f"{y:.2e}", fontsize=6))
                        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
                        plt.xlabel("Score level L")
                        plt.ylabel("Probability")
                        plt.title("P(score >= L) vs. L")
                        plt.legend()
                        plt.grid()
                        plt.savefig(str(tmp / "probability_vs_level.png"), dpi=300, bbox_inches="tight")
                        plt.close()

                    results_path = os.path.join(base_dir, "all_results.json")
                    Path(os.path.dirname(results_path)).mkdir(parents=True, exist_ok=True)
                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"Results saved to {results_path}")