# MLS-Framework

Code for **Interpretable Causal Discovery via Causal-Effect Constraints**.

This repository contains the implementation of the multilevel-splitting-based causal discovery framework used in our UAI 2026 submission, **Interpretable Causal Discovery via Causal-Effect Constraints**. The framework estimates posterior probabilities of causal-effect constraints under Bayesian causal structure learning.

## Overview

The repository supports two main experiment types:

- **Single-CE experiments**: estimate the probability of one causal-effect event, such as `P(ACE[i,j] > threshold)` or `P(ACE[i,j] < threshold)`.
- **Multi-CE experiments**: estimate the probability that several causal-effect constraints hold jointly, using a normalized score over all constraints.

The implementation includes two structure proposal kernels:

- **Structure_MCMC**: a baseline structure/weight MCMC kernel.
- **PARNI**: an informed DAG proposal kernel based on data-informed candidate neighborhoods.

## Repository Structure

```text
MLS-Framework/
├── README.md
├── requirements.txt
├── run.py                         # Single-CE experiment entry point
├── run_multi.py                   # Multi-CE experiment entry point
├── mls_frame_single_ce.py         # MLS sampler for single causal-effect constraints
├── mls_frame_multi_ce.py          # MLS sampler for multiple causal-effect constraints
├── configs/
│   └── multi_ce_constraints.py    # Named multi-CE constraint sets
└── src/
    ├── bge.py                     # BGe score implementation
    ├── helper_func.py             # Causal-effect and sampling utilities
    ├── parni_dag.py               # PARNI-DAG proposal utilities
    └── PC_skeleton.py             # PC-skeleton helper used by PARNI
```

## Installation and Environment Setup

We recommend using **Python 3.10 or later**.

### 1. Clone the repository

```bash
git clone https://github.com/ZCX031116/MLS-Framework.git
cd MLS-Framework
```

### 2. Create the environment

Using Conda:

```bash
conda create -n mls-framework python=3.10 -y
conda activate mls-framework
pip install -r requirements.txt
```

Or using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

The scripts expect data files under the `data/` directory. Large data files are not included in this repository.

### Synthetic data layout

For synthetic experiments, use the following directory structure:

```text
data/
└── mean=0/
    └── d=<num_nodes>/
        ├── target.npy
        ├── target_value.npy
        └── case<case_id>/
            ├── G_<num_nodes>Nodes_train_size_<train_size>.npy
            ├── B_<num_nodes>Nodes_train_size_<train_size>.npy
            └── train_<num_nodes>Nodes_train_size_<train_size>.npy
```

For example, for `d=4`, `case=1`, and `train_size=1000`, the expected files are:

```text
data/mean=0/d=4/target.npy
data/mean=0/d=4/target_value.npy
data/mean=0/d=4/case1/G_4Nodes_train_size_1000.npy
data/mean=0/d=4/case1/B_4Nodes_train_size_1000.npy
data/mean=0/d=4/case1/train_4Nodes_train_size_1000.npy
```

### Sachs data layout

For Sachs experiments, use:

```text
data/
└── sachs/
    ├── sachs_graph.npy
    └── sachs_data.npy
```

For Sachs, the returned `B` object is a placeholder zero matrix so that data loading has the same return signature as synthetic data: `G, B, X_train`.

## Running Experiments

Run all commands from the repository root.

### Multi-CE synthetic experiment

```bash
python3 run_multi.py \
  --dataset synthetic \
  --d-list 4,8 \
  --cases 1 \
  --constraint-names weak \
  --kernels PARNI \
  --run-num 3 \
  --seed 123
```

### Multi-CE Sachs experiment

```bash
python3 run_multi.py \
  --dataset sachs \
  --d-list 11 \
  --constraint-names sachs \
  --kernels PARNI \
  --run-num 3 \
  --seed 123
```

### Single-CE synthetic experiment

```bash
python3 run.py \
  --dataset synthetic \
  --d-list 4 \
  --cases 1 \
  --kernels Structure_MCMC \
  --run-num 3 \
  --seed 123
```

## Common Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--dataset` | Dataset to use: `synthetic` or `sachs`. | `synthetic` for `run.py`, `sachs` for `run_multi.py` |
| `--d-list` | Comma-separated graph sizes. | `4,8,16` for `run.py`; `11` for `run_multi.py` |
| `--cases` | Comma-separated synthetic case IDs. | `1` |
| `--kernels` | Comma-separated structure kernels: `Structure_MCMC`, `PARNI`. | script-dependent |
| `--n` | Number of MLS particles/samples per level. | `200` |
| `--mcmc-iterations` | Number of MCMC iterations per mutation step. | `2000` |
| `--run-num` | Number of repeated runs. | `1` |
| `--max-outer-iter` | Maximum number of adaptive MLS levels. | `10` |
| `--seed` | Seed used to generate per-run random seeds. | `None` |
| `--results-dir` | Directory for output files. | `results` |
| `--data-dir` | Directory containing input data. | `data` |

Additional `run_multi.py` arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `--constraint-names` | Named multi-CE constraint sets to run, such as `weak`, `strong`, or `sachs`. | all sets for the selected `d` |
| `--x-levels` | Comma-separated score levels. The event is `score >= L`. | `0.0` |
| `--params-per-graph` | Number of posterior weight samples per graph for approximate CE logging. | `500` |

## Outputs

Experiment outputs are written under `results/` by default.

Examples:

```text
results/PARNI/multi-CE_weak/d=4/case1/run_1/
results/PARNI/multi-CE_weak/d=4/case1/all_results.json
results/Structure_MCMC/single-CE_synthetic/d=4/case1/run_1/
```
The `results/` directory is ignored by Git.

## Notes

- The implementation assumes linear-Gaussian structural equation models and uses BGe scores for Bayesian structure learning.
- The current multi-CE constraint presets are defined in `configs/multi_ce_constraints.py`.
- PARNI uses a data-informed proposal context and is intended for data-conditioned runs.
