import random
import numpy as np
import networkx as nx
import scipy.stats as st

def pairwise_linear_ce(edge_weights):
    """Returns the pairwise causal effect given the matrix of edge weights.

    Args:
        edge_weights (np.array): (d, d) Weights of the linear model

    Returns:
        effects (np.array): (d, d) matrix of pairwise causal effects
    """
    d = edge_weights.shape[0]
    effects = np.linalg.inv(np.eye(d) - edge_weights) 
    return effects

def ce_ij(W, i, j):
    d = W.shape[0]
    A = np.eye(d) - W
    e = np.zeros(d)
    e[j] = 1.0
    x = np.linalg.solve(A, e)  
    return x[i]

def pairwise_linear_ce_no_params(g_samples, data, bge_model, params_per_graph=10,avg=True, return_B=False, R = None):
    """
    Returns the pairwise (linear) causal effect, averaged over the DAG samples.
    """
    if R is None:
        R = bge_model.calc_R(data)
    N, d = data.shape
    B = [[] for _ in range(d)]
    cnt = 0
    for G_sample in g_samples:
        for i in range(d):
            parents_mask = G_sample[:, i].astype(bool)
            if np.any(parents_mask):
                l = np.sum(parents_mask) + 1
                parents_child_mask = np.copy(parents_mask)
                parents_child_mask[i] = True
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
                bs = np.expand_dims(dist.rvs(params_per_graph), axis=0) if params_per_graph == 1 else dist.rvs(params_per_graph)
                for b in bs:
                    column = np.zeros(d)
                    column[parents_mask] = b
                    B[i].append(column)
            else:
                for _ in range(params_per_graph):
                    B[i].append(np.zeros(d))
    
    B = np.array(B)  # (d-col, num_total_samples, d-row)
    B = np.swapaxes(np.swapaxes(B, 0, 1), 1, 2)
    effects = [np.linalg.inv(np.eye(d) - B_sample) for B_sample in B]
    avg_effects = np.mean(np.array(effects), axis=0)
    if return_B:
        if avg:
            return B, avg_effects
        else:
            return B, effects
    else:
        if avg:
            return avg_effects
        else:
            return effects

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

def build_H_topk_corr(data, K):
    data = np.asarray(data, dtype=float)
    d = data.shape[1]
    K = int(min(max(K, 1), d - 1))
    C = np.abs(np.corrcoef(data, rowvar=False))
    np.fill_diagonal(C, 0.0)
    H = np.zeros((d, d), dtype=int)
    for v in range(d):
        cand = np.argsort(-C[:, v])[:K]
        H[cand, v] = 1
    # symmetrize to avoid directional starvation
    H = np.maximum(H, H.T)
    np.fill_diagonal(H, 0)
    return H
    
def p_structure_schedule(d, T=4000, min_weight_moves=600) -> float:
    import math
    p = 0.60 + 0.10 * math.log2(d / 8.0)
    p = max(0.50, min(0.85, p))
    p = min(p, 1.0 - min_weight_moves / max(T, 1))
    return float(max(0.50, min(0.85, p)))

def check_acyclic(adj_matrix):
    A = np.asarray(adj_matrix)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"adj_matrix must be a square 2D array, got shape={A.shape}")
    d = A.shape[0]
    indeg = (A != 0).sum(axis=0).astype(np.int64)
    stack = [i for i in range(d) if indeg[i] == 0]
    visited = 0
    while stack:
        u = stack.pop()
        visited += 1
        row = A[u]
        vs = np.nonzero(row != 0)[0]
        for v in vs:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return visited == d

def build_reachability(adj):
    """Floyd–Warshall transitive closure, returns a boolean matrix reach[i, j] indicating if i→j is reachable."""
    reach = adj.astype(bool).copy()
    d = adj.shape[0]
    for k in range(d):
        reach |= np.outer(reach[:, k], reach[k, :])
    return reach

def random_dag_topo(num_nodes, p = 0.3):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < p:
                G.add_edge(nodes[i], nodes[j])
    return G

def sample_random_graphs(n, d, p = 0.3,size_switch = 24):
    graphs = [nx.to_numpy_array(random_dag_topo(d, p)) for _ in range(n)]
    return graphs
