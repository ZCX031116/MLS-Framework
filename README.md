# MLS-Framework

Code for **Interpretable Causal Discovery via Causal-Effect Constraints**.

## Installation and Environment Setup

This repository contains the implementation of the MLS-based causal discovery framework used in our UAI 2026 submission, **Interpretable Causal Discovery via Causal-Effect Constraints**.

The code is written in Python and depends on common scientific-computing libraries such as NumPy, SciPy, NetworkX, tqdm, joblib, and tqdm-joblib.

We recommend using **Python 3.10 or later**.

### 1. Clone the repository

### 2. Environment setting

```bash
cd MLS-Framwork
conda create -n mls-framework python=3.10 -y
conda activate mls-framework
pip install -r requirements.txt
```

### 3. Run
```bash
python3 run_multi.py 
--dataset synthetic 
--d-list 4,8 
--cases 1 
--constraint-names weak 
--kernels PARNI 
--run-num 3
```
