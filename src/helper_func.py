import random
import numpy as np
import networkx as nx
import scipy.stats as st


def pairwise_linear_ce(edge_weights):
    """Return the pairwise linear causal effects for a weighted DAG."""
    d = edge_weights.shape[0]
    return np.linalg.inv(np.eye(d) - edge_weights)


def ce_ij(W, i, j):
    d = W.shape[0]
    A = np.eye(d) - W
    e = np.zeros(d)
    e[j] = 1.0
    x = np.linalg.solve(A, e)
    return x[i]


def pairwise_linear_ce_no_params(
    g_samples,
    data,
    bge_model,
    params_per_graph=10,
    avg=True,
    return_B=False,
    R=None,
    rng=None,
):
    """Sample edge weights from the BGe posterior and compute linear effects.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random generator used by SciPy's multivariate-t sampler.  When omitted,
        SciPy falls back to NumPy's legacy global RNG, so callers can still make
        this function reproducible by calling ``np.random.seed(seed)``.
    """
    if R is None:
        R = bge_model.calc_R(data)
    N, d = data.shape
    B = [[] for _ in range(d)]
    for G_sample in g_samples:
        for i in range(d):
            parents_mask = G_sample[:, i].astype(bool)
            if np.any(parents_mask):
                l = np.sum(parents_mask) + 1
                R22 = R[i, i]
                R12 = R[parents_mask, i]
                R21 = R[i, parents_mask]
                R11 = R[parents_mask, :][:, parents_mask]
                loc = np.linalg.inv(R11) @ R12
                deg_free = bge_model.alpha_w + N - d + l
                shape = np.linalg.inv(
                    deg_free /
                    (R22 - R21 @ np.linalg.inv(R11) @ R12
                     ) *
                    R11
                )
                dist = st.multivariate_t(loc=loc, shape=shape, df=deg_free)
                bs = (
                    np.expand_dims(
                        dist.rvs(params_per_graph, random_state=rng), axis=0
                    )
                    if params_per_graph == 1
                    else dist.rvs(params_per_graph, random_state=rng)
                )
                for b in bs:
                    column = np.zeros(d)
                    column[parents_mask] = b
                    B[i].append(column)
            else:
                for _ in range(params_per_graph):
                    B[i].append(np.zeros(d))
    B = np.array(B)
    B = np.swapaxes(np.swapaxes(B, 0, 1), 1, 2)
    effects = [np.linalg.inv(np.eye(d) - B_sample) for B_sample in B]
    avg_effects = np.mean(np.array(effects), axis=0)
    if return_B:
        return B, avg_effects if avg else effects
    return avg_effects if avg else effects


def log_and_print(message, file=None, console_output=True):
    if console_output:
        print(message)
    if file is not None:
        file.write(str(message) + "\n")


def get_erdos_renyi_q(d, edges_per_node):
    max_edges = d * (d - 1) / 2
    q = (edges_per_node * d) / max_edges
    return min(q, 0.5)


def get_p_edge_for_inference(d, edges_per_node):
    q = get_erdos_renyi_q(d, edges_per_node)
    return 0.5 * q


def p_structure_schedule(d, T=4000, min_weight_moves=600) -> float:
    import math
    p = 0.60 + 0.10 * math.log2(d / 8.0)
    p = max(0.50, min(0.85, p))
    p = min(p, 1.0 - min_weight_moves / max(T, 1))
    return float(max(0.50, min(0.85, p)))


def random_dag_topo(num_nodes, p=0.3, rng=None):
    """Sample a random DAG from a random topological order.

    When ``rng`` is omitted, this uses Python's global ``random`` module, so
    callers can reproduce samples by calling ``random.seed(seed)``.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    nodes = list(range(num_nodes))

    if rng is None:
        random.shuffle(nodes)
        random_uniform = random.random
    else:
        nodes = list(rng.permutation(nodes))
        random_uniform = rng.random

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random_uniform() < p:
                G.add_edge(nodes[i], nodes[j])
    return G


def sample_random_graphs(n, d, p=0.3, rng=None):
    return [nx.to_numpy_array(random_dag_topo(d, p, rng=rng)) for _ in range(n)]
