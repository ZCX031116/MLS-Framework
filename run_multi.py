import os
import json
import random
import numpy as np
from src.bge import BGe
from mls_frame_multi_ce import Multilevel_Multi_CE
from configs.multi_ce_constraints import get_multi_ce_constraints
from src.helper_func import (
    pairwise_linear_ce_no_params, 
    p_structure_schedule,
    log_and_print,
)
from pathlib import Path

dataset = "sachs"      # "sachs" or "synthetic"
mean = 0
X_levels = [0.0]
n = 200
mcmc_iterations = 2000
Run_Num = 1

for d in [11]:        
    ce_constraints_list = get_multi_ce_constraints(d)

    for name, ce_constraints in ce_constraints_list:
        for structure_kernel in ["PARNI", "Structure_MCMC"]:
            results = []
            for num in [1]:
                for run_num in range(1,Run_Num+1):
                    seed = random.randint(0, 10000)
                    rng = np.random.default_rng(seed)
                    bge_model = BGe(d=d, alpha_u=10)
                    if dataset == "synthetic":
                        save_dir = f"results/{structure_kernel}/multi-CE_{name}/d={d}/case{num}/run_{run_num}"
                        load_dir = f"data/mean=0/d={d}/"
                        load_case_dir = load_dir + f"case{num}"
                        G = np.load(f"{load_case_dir}/G_{d}Nodes_train_size_1000.npy")
                        X_train = np.load(f"{load_case_dir}/train_{d}Nodes_train_size_1000.npy")

                    elif dataset == "sachs":
                        save_dir = f"results/multi-CE_sachs/{structure_kernel}/run_{run_num}"
                        load_dir = "data/sachs/"
                        G = np.load(f"{load_dir}/sachs_graph.npy")
                        X_train = np.load(f"{load_dir}/sachs_data.npy")
                    else:
                        raise ValueError(f"Unknown dataset={dataset!r}")
                    
                    ce = pairwise_linear_ce_no_params(
                            np.copy([G]), X_train, bge_model, params_per_graph=500, avg=True, return_B=False
                        )

                    for (src, end, op, thr, sc) in ce_constraints:
                        print(f"approx CE({src}->{end}) = {ce[int(src), int(end)]:.6g}  constraint: CE {op} {thr}")

                    tmp = Path(save_dir)
                    tmp.mkdir(parents=True, exist_ok=True)
                    output_file = os.path.join(tmp, f"{structure_kernel}_run{run_num}_output_results.txt")

                    multilevel_model = Multilevel_Multi_CE(
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
                        for (src, end, op, thr, sc) in ce_constraints:
                            log_and_print(f"approx CE({src}->{end}) = {ce[int(src), int(end)]:.6g}", f)

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

                results_path = tmp.parent / "all_results.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {results_path}")
