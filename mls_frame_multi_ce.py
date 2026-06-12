from pathlib import Path
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import scipy.stats as st
from collections import deque
import src.parni_dag as parni
from src.PC_skeleton import build_H_pc_plus
from src.parni_dag import(
    parni_prepare_context,
    parni_make_LA_from_G,
    parni_step_one,
    parni_update_pips_eq9,
)
from src.helper_func import (
    ce_ij, 
    pairwise_linear_ce_no_params, 
    log_and_print,
    get_p_edge_for_inference,
    sample_random_graphs,
)

class Multilevel_Multi_CE:
    def __init__(self, bge_model, data, X, ce_constraints, max_outer_iter, save_dir=None, output_file=None,
                    edges_per_node = 2, params_per_graph=50, rng=None, structure_kernel = "Structure_MCMC", p_structure = 0.5,
                    save_level_samples: bool = False, save_level_weights: bool = False, save_level_all: bool = False,
                    samples_dirname = "level_samples", verbose: bool = False):
        d = data.shape[1]
        self.bge_model = bge_model
        self.R = bge_model.calc_R(data)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.data = data
        self.ce_constraints = self._prepare_ce_constraints(ce_constraints)
        self._ce_signs = np.asarray([c["sign"] for c in self.ce_constraints], dtype=float)
        self._ce_thr_metric = np.asarray([c["thr_metric"] for c in self.ce_constraints], dtype=float)
        self._ce_scales = np.asarray([c["scale"] for c in self.ce_constraints], dtype=float)
        self.X = self._prepare_score_targets(X)
        self.i = int(self.ce_constraints[0]["i"])
        self.j = int(self.ce_constraints[0]["j"])
        self.sigma = 1.0
        self.mu = 0.0
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.output_file = output_file
        self.verbose = bool(verbose)
        self.p_edge = get_p_edge_for_inference(d, edges_per_node)
        self.params_per_graph = params_per_graph
        self.max_outer_iter = max_outer_iter
        self.structure_kernel = structure_kernel
        self.p_structure = p_structure
        self.save_level_samples = bool(save_level_samples)
        self.save_level_weights = bool(save_level_weights)
        self.save_level_all = bool(save_level_all)
        self.samples_dirname = str(samples_dirname)
        self.level_samples_dir = None
        if self.save_level_samples:
            if self.save_dir is None:
                raise ValueError("save_dir is required when save_level_samples=True.")
            self.level_samples_dir = self.save_dir / self.samples_dirname
            self.level_samples_dir.mkdir(parents=True, exist_ok=True)

        if self.structure_kernel == "PARNI":
            if self.verbose:
                print("Preparing PARNI context...")
            X_p_n = self.data.T
            d = self.data.shape[1]
            H, extra_parents = build_H_pc_plus(
                self.data,
                alpha=0.1,
                max_cond_set=6,
                candidate_cap=12,
                extend_one=True,
                extra_parent_cap=8,
                verbose=False,
            )
            self.parni_ctx = parni_prepare_context(
                X_p_n=X_p_n,
                h=self.p_edge,
                bge_obj=self.bge_model,
                H=H,
                kappa=0.1,      
                omega=0.5,       
                pips_mode="uniform",
                pips_in=0.5,
                pips_out=0.5,
                extra_parents=extra_parents,
            )
            self.parni_ctx["omega_N_tilde"] = 10 if d <= 16 else 20
            self.parni_ctx["omega_adapt"] = True
            self.parni_ctx["pips_adapt"] = False
            self.parni_ctx["pips_eps"] = 0.05

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


    def _clone_parni_ctx_for_chain(self, base_ctx, burn_in=None):
        ctx = dict(base_ctx)
        if "hp" in base_ctx:
            ctx["hp"] = base_ctx["hp"]
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
            b_i = weight_matrix[parents, i]
            try:
                dist = st.multivariate_t(loc=loc, shape=shape_matrix, df=deg_free)
                log_p += dist.logpdf(b_i)
            except np.linalg.LinAlgError:
                return -np.inf
        return log_p

    def log_posterior_with_weights(self, adjacency_matrix, weight_matrix, mll_score=None):
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

    def initialize_edge_weight_matrix(self, adjacency_matrix):
        G_batch = np.array(adjacency_matrix)
        Bs, _ = pairwise_linear_ce_no_params(G_batch, self.data, self.bge_model,
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
            pa_G = np.where(G[:, v])[0]
            pa_Gp = np.where(Gp[:, v])[0]
            if pa_G.shape[0] != pa_Gp.shape[0] or (pa_G.shape[0] > 0 and not np.array_equal(pa_G, pa_Gp)):
                changed.append(v)
        return changed

    def _refresh_weights_S1(self, G: np.ndarray, W: np.ndarray, Gp: np.ndarray, rng):
        G = np.asarray(G, dtype=int)
        Gp = np.asarray(Gp, dtype=int)
        W = np.asarray(W, dtype=float)
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
            u, v = v, u
        uv = int(A[u, v] != 0)
        vu = int(A[v, u] != 0)

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

        if proposed_state == 0:
            new_adj = A.copy()
            new_adj[u, v] = 0
            new_adj[v, u] = 0
            return new_adj, 0.0, 0.0

        if proposed_state == 1:
            if current_state == 2:
                would_cycle = _has_path(v, u, ignore_edge=(v, u))
            else:
                would_cycle = _has_path(v, u, ignore_edge=None)
            if would_cycle:
                return A.copy(), 0.0, 0.0
            new_adj = A.copy()
            new_adj[u, v] = 1
            new_adj[v, u] = 0
            return new_adj, 0.0, 0.0

        else:
            if current_state == 1:
                would_cycle = _has_path(u, v, ignore_edge=(u, v))
            else:
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
        new_w = np.asarray(weight_matrix, dtype=float).copy()
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
            new_adj, log_qG_fwd, log_qG_rev = self.propose_new_structure_only(adjacency_matrix, rng=rng)
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
                    rng=None):
        if rng is None:
            rng = np.random.default_rng()

        current_adj = np.array(initial_adj, copy=True)
        current_w   = np.array(initial_w, copy=True)

        num_edges = float(np.sum(current_adj))
        logit_p = np.log(self.p_edge / (1.0 - self.p_edge))
        log_prior_g = num_edges * logit_p
        if np.isneginf(log_prior_g) or np.isnan(log_prior_g):
            log_prior_g = -100.0
        mll_score = self.bge_model.mll(current_adj, self.data)
        current_log_graph = log_prior_g + mll_score
        p_G = log_prior_g
        p_X_G = mll_score

        ACC = 0
        ACC_weight = 0
        ACC_structure = 0
        weight_moves = 0
        structure_moves = 0
        parni_ctx = None
        if self.structure_kernel == "PARNI":
            parni_ctx = self._clone_parni_ctx_for_chain(self.parni_ctx, burn_in=int(burn_in))
            LA = parni_make_LA_from_G(current_adj.astype(int), parni_ctx)

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
                        log_prior_g = p_G
                        mll_score = p_X_G
                        proposed_log_graph = current_log_graph
                    else:
                        num_edges = float(np.sum(new_adj))
                        logit_p = np.log(self.p_edge / (1.0 - self.p_edge))
                        log_prior_g = num_edges * logit_p
                        if np.isneginf(log_prior_g) or np.isnan(log_prior_g):
                            log_prior_g = -100.0
                        mll_score = self.bge_model.mll(new_adj, self.data)
                        proposed_log_graph = log_prior_g + mll_score
                        log_acceptance_ratio = (proposed_log_graph + logp_cur) - (current_log_graph + logp_prop)
                        acceptance_ratio = 1.0 if log_acceptance_ratio >= 0 else float(np.exp(log_acceptance_ratio))

            elif self.structure_kernel == "PARNI":
                LA_prev = LA
                new_adj, new_w, logp_prop, logp_cur, LA_prop,move = self.propose_new_state_PARNI(
                    LA, current_w, ctx=parni_ctx, rng=rng, p_structure=self.p_structure
                )
                move_type = move 
                LA_prop_llh = float(LA_prop.llh)
                pairwise_effect = self._splitting_metric(new_w)
                if pairwise_effect < level:
                    acceptance_ratio = 0.0
                    LA = LA_prev
                else:
                    if move_type == "weight":
                        acceptance_ratio = 1.0
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

            if rng.random() < acceptance_ratio:
                current_adj = new_adj
                current_w   = new_w
                current_log_graph = proposed_log_graph
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
            if self.structure_kernel == "PARNI" and move == "structure":
                parni_update_pips_eq9(parni_ctx, LA.curr)

        acc_rate = ACC / float(iterations)
        acc_structure_rate = ACC_structure / float(structure_moves) if structure_moves > 0 else 0.0
        acc_weight_rate    = ACC_weight    / float(weight_moves)    if weight_moves    > 0 else 0.0

        return current_adj, current_w, acc_rate, acc_structure_rate, acc_weight_rate

    def compute_sample_graph_parallel(self, G, W, mcmc_iterations, level, seed=None):
        """
        Each parallel job constructs its own RNG and passes it down.
        MCMC stage will NOT use self.rng.
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        current_adj, current_w, acc_rate, acc_structure_rate, acc_weight_rate = self.mcmc_sampling(
            initial_adj=G,
            initial_w=W,
            iterations=mcmc_iterations,
            burn_in=int(mcmc_iterations * 0.1),
            level=level,
            rng=rng,
        )
        return current_adj, current_w, acc_rate, acc_structure_rate, acc_weight_rate

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

    def _log(self, message: str) -> None:
        if self.output_file is not None:
            with open(self.output_file, "a") as f:
                log_and_print(message, f, console_output=self.verbose)
        elif self.verbose:
            print(message)

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
            if self.save_dir is None:
                raise ValueError("save_dir is required when save_level_samples=True.")
            out_dir = self.save_dir / "level_samples"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.level_samples_dir = out_dir
        it = int(iteration)
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
        Estimate log probabilities for the configured multi-CE score levels.

        The public API is quiet by default: this method returns the log
        probability estimates and does not write diagnostics or figures unless
        explicit optional sample saving was enabled at construction time.
        """
        self._log("Starting adaptive leveling...")
        self._log(f"(T={mcmc_iterations}, n param={n}) (kernel method: {self.structure_kernel})")
        probability_list = []
        S = 0  
        iteration = 0
        idx_target = 0
        current_level = -np.inf
        G_samples = sample_random_graphs(n, self.data.shape[1], p=self.p_edge)
        W_samples = list(self.initialize_edge_weight_matrix(G_samples))
        while True:
            if (self.max_outer_iter is not None) and (iteration >= self.max_outer_iter):
                remaining = len(self.X) - idx_target
                probability_list.extend([-50] * remaining)
                self._log("Reached max_outer_iter; returning floor log probabilities for unresolved targets.")
                return probability_list
            self._log(f"Current level (metric)={current_level} | {self._condition_str(current_level)}")
            iteration += 1
            seeds = self.rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32)

            with tqdm_joblib(tqdm(desc="Parallel ACE Computation", total=n, disable=not self.verbose)):
                results = Parallel(n_jobs=-1)(
                    delayed(self.compute_sample_graph_parallel)(G, W, mcmc_iterations, current_level, int(sd))
                    for (G, W, sd) in zip(G_samples, W_samples, seeds)
                )
            rho = 0.1
            k = int(np.ceil(rho * n))
            k = min(max(k, 1), n)  
            new_adjs = [r[0] for r in results]
            new_ws = [r[1] for r in results]
            acc_rates = [r[2] for r in results]
            acc_structure_rates = [r[3] for r in results]
            acc_weight_rates = [r[4] for r in results]
            self._log(f"Average acceptance rate in this iteration: {np.mean(acc_rates)}")
            self._log(f"Average structure acceptance rate in this iteration: {np.mean(acc_structure_rates)}")
            self._log(f"Average weight acceptance rate in this iteration: {np.mean(acc_weight_rates)}")

            graph_pairs = []
            metric_details = [self._splitting_metric_details(w) for w in new_ws]
            score_list = [md[0] for md in metric_details]
            metric_vec_list = [md[2] for md in metric_details]
            for adj, w, sc, mvec in zip(new_adjs, new_ws, score_list, metric_vec_list):
                graph_pairs.append([sc, adj, w, mvec])
            sorted_graph_pairs = sorted(graph_pairs, key=lambda x: x[0], reverse=True)
            current_level = sorted_graph_pairs[k - 1][0]

            self._log(
                f"Next level selected at rho={rho}: level(metric)={current_level}, "
                f"k={k} | {self._condition_str(current_level)}"
            )
            proportion = sum(1 for item in sorted_graph_pairs if item[0] >= current_level) / len(sorted_graph_pairs)
            if proportion == 0:
                return probability_list + [-50.0] * (len(self.X) - idx_target)
            surv_items = [item for item in sorted_graph_pairs if item[0] >= current_level]
            new_G_samples = [item[1] for item in surv_items]
            new_W_samples = [item[2] for item in surv_items]
            ace_surv_sorted = [item[0] for item in surv_items]
            if getattr(self, "save_level_samples", False):
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
                        ace=score_list,
                        kind="all",
                    )
            self._log(f"Level {current_level} survivor proportion: {proportion}")

            last_valid_idx = None
            for i, target_value in enumerate(self.X[idx_target:], start=idx_target):
                if current_level >= target_value:
                    last_valid_idx = i
                    count_exceed = sum(1 for item in sorted_graph_pairs if item[0] > target_value)
                    final_proportion = count_exceed / n
                    self._log(
                        f"Level {current_level} reaches target {self._condition_str(target_value)}; "
                        f"final proportion={final_proportion}"
                    )
                    if final_proportion == 0:
                        log_probability = -50.0
                    else:
                        log_probability = S + np.log(final_proportion)
                    probability_list.append(log_probability)
            if last_valid_idx is not None:
                idx_target = last_valid_idx+1 
                if idx_target == len(self.X):
                    return probability_list
            S += np.log(proportion)
            G_samples, W_samples, _ = self._resample_balanced(new_G_samples, new_W_samples, n)
            self._log(f"Iteration {iteration} completed. Current log S: {S:.4f}")
        return np.exp(S)
