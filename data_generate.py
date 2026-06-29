import os
import numpy as np
import random 
from numpy.random import default_rng

def make_erdosrenyi_graph(d, edges_per_node, seed=None):
    """Generates a random Erdos-Renyi directed acyclic graph.
    Args:
        d (int): Number of nodes in graph
        edges_per_node (int): Expected number of edges per node
        seed (int): Random seed

    Returns:
        G (np.array): (d, d) adjacency array representing the random graph
    """
    rng = default_rng(seed=seed)
    p = min((edges_per_node * d) / (d * (d - 1) / 2), 0.5)
    G_array = rng.binomial(n=1, p=p, size=(d, d))
    # Make acyclic
    G_array = np.tril(G_array, k=-1)
    # Randomly permute nodes
    P = rng.permutation(np.eye(d))
    G_array_permuted = P.T @ G_array @ P
    return G_array_permuted

def make_linear_model(G, rng, weight_mean=0.0, weight_sd=1.0):
    """Given a DAG representing the graphical model, generate random weights of a linear model for each edge in the DAG.

    Args:
        G (np.array): (d, d) array representing the adjacency matrix
        rng: NumPy Generator
        weight_mean (float): Mean weight for edges
        weight_sd (float): Standard deviation of weight for edges

    Returns:
        weights: (d, d) array of weights
    """

    B = rng.normal(weight_mean, weight_sd, size=G.shape)
    B_masked = B * G
    return B_masked

def generate_linear_data(n_samples, B, rng, noise_sd=0.316):
    """Generates data from a Linear Gaussian model.

    Args:
        n_samples (int): Number of samples of data to generate
        B (np.array): (d, d) array of edge weights
        rng: NumPy Generator
        noise_sd (float/np.array): Either a scalar or a 1d array of noise standard deviations for each variable

    Returns:
         data (np.array): (n, d) representing the samples
    """
    d = B.shape[0]
    eps = rng.normal(loc=0, scale=noise_sd, size=(n_samples, d))
    return eps @ np.linalg.inv(np.eye(d) - B)

if __name__ == "__main__":
    train_size = 1000
    d = 4
    edges_per_node = 2 # how dense of the graph
    test_size = 100
    weight_mean = 2

    seed = np.random.randint(0, 100)
    rng = np.random.default_rng(seed)
    name = f"{d}Nodes"
    num = 2
    type = "mean=0" # or "sachs"

    load_dir = f"data_test/{type}/d={d}/case{num}"
    os.makedirs(load_dir, exist_ok=True)
    
    G = make_erdosrenyi_graph(d=d, edges_per_node=edges_per_node, seed=rng)
    B = make_linear_model(G, rng, weight_mean=weight_mean)
    np.save(load_dir + f"/G_{name}_train_size_1000.npy", G)
    np.save(load_dir + f"/B_{name}_train_size_1000.npy", B)
    X_train = generate_linear_data(train_size, B, rng)
    X_test = generate_linear_data(test_size, B, rng)
    np.save(load_dir + f"/train_{name}_train_size_1000.npy", X_train)
    np.save(load_dir + f"/test_{name}_train_size_1000.npy", X_test)
    print("G", G)
    print("B", B)
    print("X_train", X_train)
    true_effects = np.linalg.inv(np.eye(d) - B)
    print("true_effects", true_effects)
