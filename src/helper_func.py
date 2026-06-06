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
import json
from adjustText import adjust_text
from src.bge import BGe

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

def plot_edge_frequency_and_weight_avg_graphs(ce_samples, gs_np, thetas_np, i, j, threshold, save_path, file_name, condition_label=None):
    """Plot edge frequency and average-weight graphs for CE[i, j] > threshold."""
    d = gs_np.shape[1]
    edge_count = np.zeros((d, d))
    edge_weight_sum = np.zeros((d, d))

    count = 0  # Number of valid samples

    for G_sample, theta_sample, ce in zip(gs_np, thetas_np, ce_samples):
        if not np.isfinite(ce):
            continue
        if ce <= threshold:
            continue
        count += 1
        for src in range(d):
            for tgt in range(d):
                if G_sample[src, tgt] == 1:
                    edge_count[src, tgt] += 1
                    edge_weight_sum[src, tgt] += theta_sample[src, tgt]
    with open(file_name, "a") as f:
        _lbl = condition_label if condition_label is not None else f"CE[{i},{j}] > {threshold}"
        log_and_print(f"{count} samples where {_lbl}", f)

    # Normalize frequency to [0, 1]
    edge_freq = edge_count / count if count > 0 else edge_count

    # Average edge weight where count > 0
    edge_weight_avg = np.zeros_like(edge_weight_sum)
    np.divide(edge_weight_sum, edge_count, out=edge_weight_avg, where=edge_count > 0)

    # Plot Edge Frequency Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(edge_freq, cmap='Reds', interpolation='nearest')
    _lbl = condition_label if condition_label is not None else f"CE[{i},{j}] > {threshold}"
    plt.title(f"Edge Frequency ({_lbl})")
    plt.colorbar(label='Edge Appearance Frequency')
    plt.xlabel("Target Node")
    plt.ylabel("Source Node")
    plt.savefig(save_path / f"edge_frequency_CE_{i}_{j}_gt_{threshold}.png", dpi=300)
    plt.close()

    # Plot Average Edge Weight Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(edge_weight_avg, cmap='coolwarm', interpolation='nearest')
    _lbl = condition_label if condition_label is not None else f"CE[{i},{j}] > {threshold}"
    plt.title(f"Average Edge Weight ({_lbl})")
    plt.colorbar(label='Average Edge Weight')
    plt.xlabel("Target Node")
    plt.ylabel("Source Node")
    plt.savefig(save_path / f"edge_weight_avg_CE_{i}_{j}_gt_{threshold}.png", dpi=300)
    plt.close()

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

def level_tag(level):
    try:
        lv = float(level)
    except Exception:
        return "nan"
    if not np.isfinite(lv):
        return "nan"
    s = f"{lv:.6e}"
    return s.replace("+", "")

def atomic_json_dump(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def summarize_array(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [0, 10, 50, 90, 95, 99, 100])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(q[0]),
        "q10": float(q[1]),
        "q50": float(q[2]),
        "q90": float(q[3]),
        "q95": float(q[4]),
        "q99": float(q[5]),
        "max": float(q[6]),
    }

def draw_graphs_in_grid(output_file, save_dir, G_samples, ace_values, iteration, level, n_samples=9):
    selected_indices = random.sample(range(len(G_samples)), min(n_samples, len(G_samples)))
    selected_graphs = [G_samples[i] for i in selected_indices]
    selected_ace_values = [ace_values[i] for i in selected_indices]
    fig, axes = plt.subplots(3, 3, figsize=(36, 36))
    axes = axes.flatten()
    for idx, (G, ace, ax) in enumerate(zip(selected_graphs, selected_ace_values, axes)):
        graph = nx.from_numpy_array(G, create_using=nx.DiGraph())
        is_acyclic = nx.is_directed_acyclic_graph(graph)
        title = f"ACE: {ace:.4f}"
        if not is_acyclic:
            title += " (Contains Cycle)"
        pos = nx.kamada_kawai_layout(graph)
        nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_color='lightblue',
                font_size=10, node_size=500, arrows=True)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.suptitle(f"Graphs at Iteration {iteration} (Level {level})", fontsize=16)
    output_file = f"{save_dir}/graphs_iteration_{iteration}_level_{level}.png"
    plt.savefig(output_file)
    plt.close()
    with open(output_file, "a") as f:
        log_and_print(f"Graphs saved to {output_file}", f)

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

def adj_hash(adj: np.ndarray) -> str:
    """Stable hash for an adjacency matrix (for uniqueness diagnostics)."""
    a = np.asarray(adj, dtype=np.uint8)
    return hashlib.sha1(a.tobytes()).hexdigest()

def unique_graph_count(adjs) -> int:
    seen = set()
    for a in adjs:
        seen.add(adj_hash(a))
    return len(seen)

def sanity_check_weights_matrix(G, W, eps):
    G = np.asarray(G, dtype=int)
    W = np.asarray(W, dtype=float)
    mask_nonedge = (G == 0)
    np.fill_diagonal(mask_nonedge, False)
    viol_nonedge = np.where(mask_nonedge & (np.abs(W) > eps))
    viol_diag = np.where(np.abs(np.diag(W)) > eps)[0]
    max_nonedge = 0.0
    if viol_nonedge[0].size > 0:
        max_nonedge = float(np.max(np.abs(W[viol_nonedge])))
    max_diag = 0.0
    if viol_diag.size > 0:
        max_diag = float(np.max(np.abs(np.diag(W))))
    return {
        "num_nonedge_viol": int(viol_nonedge[0].size),
        "max_nonedge_viol": max_nonedge,
        "num_diag_viol": int(viol_diag.size),
        "max_diag_viol": max_diag,
    }