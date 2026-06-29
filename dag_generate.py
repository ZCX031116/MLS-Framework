import itertools
import numpy as np
import networkx as nx

# Script for generating all DAGs for a given number of nodes. The generated DAGs are saved in a numpy array file.

def generate_all_dags(d):
    all_dags = []
    nodes = list(range(d))
    N = 0
    max_edges = d * (d - 1) // 2
    all_edges = list(itertools.product([0, 1], repeat=d * d))
    all_edges.sort(key=lambda edges: sum(edges))
    for edges in all_edges:
        current_edge_count = sum(edges)
        if current_edge_count > max_edges:
            break
        adjacency_matrix = np.array(edges).reshape(d, d)
        N = N + 1
        G = nx.DiGraph(adjacency_matrix)
        if nx.is_directed_acyclic_graph(G):
            all_dags.append(adjacency_matrix)
        else:
            pass

    n = len(all_dags)
    all_dags_matrix = np.zeros((n, d, d), dtype=int)
    for i in range(n):
        all_dags_matrix[i] = all_dags[i]

    return all_dags_matrix

if __name__ == "__main__":
    for d in range(1,6):
        name = f"{d}Nodes"
        all_dags = generate_all_dags(d)
        np.save(f"data/DAG_{name}.npy", all_dags)