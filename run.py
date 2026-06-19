import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from mls_frame_single_ce import Multilevel_Single_CE
from src.bge import BGe
from src.helper_func import p_structure_schedule, log_and_print


def _csv_ints(value):
    return [int(x) for x in value.split(",") if x.strip()]


def _csv_strings(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run single causal-effect MLS experiments.")
    parser.add_argument("--dataset", choices=["sachs", "synthetic"], default="synthetic")
    parser.add_argument("--d-list", type=_csv_ints, default=[4, 8, 16], help="Comma-separated graph sizes.")
    parser.add_argument("--cases", type=_csv_ints, default=[1], help="Comma-separated synthetic case numbers.")
    parser.add_argument(
        "--kernels",
        type=_csv_strings,
        default=["Structure_MCMC"],
        help="Comma-separated structure kernels.",
    )
    parser.add_argument("--n", type=int, default=200, help="Number of MLS samples per level.")
    parser.add_argument("--mcmc-iterations", type=int, default=2000)
    parser.add_argument("--run-num", type=int, default=1, help="Number of repeated runs.")
    parser.add_argument("--max-outer-iter", type=int, default=10)
    parser.add_argument("--alpha-u", type=float, default=10)
    parser.add_argument("--mean", type=int, default=0, help="Synthetic data mean directory.")
    parser.add_argument("--train-size", type=int, default=1000, help="Synthetic train size used in data filenames.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for generating per-run RNG seeds.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def load_data(args, d, case_num):
    if args.dataset == "synthetic":
        load_dir = args.data_dir / f"mean={args.mean}" / f"d={d}"
        load_case_dir = load_dir / f"case{case_num}"
        G = np.load(load_case_dir / f"G_{d}Nodes_train_size_{args.train_size}.npy")
        B = np.load(load_case_dir / f"B_{d}Nodes_train_size_{args.train_size}.npy")
        X_train = np.load(load_case_dir / f"train_{d}Nodes_train_size_{args.train_size}.npy")
        return load_dir, G, B, X_train

    if args.dataset == "sachs":
        load_dir = args.data_dir / "sachs"
        G = np.load(load_dir / "sachs_graph.npy")
        B = np.zeros_like(G)  # Not used for Sachs, but keeps the return structure consistent.
        X_train = np.load(load_dir / "sachs_data.npy")
        return load_dir, G, B, X_train

    raise ValueError(f"Unknown dataset={args.dataset!r}")


def load_target(load_dir, case_num):
    target = np.load(load_dir / "target.npy")[case_num - 1]
    target_value_list = np.load(load_dir / "target_value.npy")[case_num - 1]
    return target, target_value_list


def get_base_dir(args, d, structure_kernel):
    if args.dataset == "synthetic":
        return args.results_dir / structure_kernel / "single-CE_synthetic" / f"d={d}"

    return args.results_dir / structure_kernel / "single-CE_sachs"


def get_save_dir(args, d, structure_kernel, case_num, run_num):
    if args.dataset == "synthetic":
        return get_base_dir(args, d, structure_kernel) / f"case{case_num}" / f"run_{run_num}"

    return get_base_dir(args, d, structure_kernel) / f"run_{run_num}"


def get_output_file(args, save_dir, structure_kernel, case_num, run_num):
    if args.dataset == "synthetic":
        filename = f"{structure_kernel}_case{case_num}_run{run_num}_output_results.txt"
    else:
        filename = f"{structure_kernel}_run{run_num}_output_results.txt"

    return os.path.join(save_dir, filename)


def seed_run(seed):
    """Seed all RNGs that can affect initialization for one run."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def main():
    args = parse_args()
    seed_source = random.Random(args.seed)

    # d = 4, 8, 16, 32 for synthetic dataset; d = 11 for Sachs dataset.
    for d in args.d_list:
        for structure_kernel in args.kernels:
            results = []
            base_dir = get_base_dir(args, d, structure_kernel)

            for run_num in range(1, args.run_num + 1):
                run_results = []

                for case_num in args.cases:
                    seed = seed_source.randint(0, 10000)
                    rng = seed_run(seed)
                    bge_model = BGe(d=d, alpha_u=args.alpha_u)
                    load_dir, G, B, X_train = load_data(args, d, case_num)
                    target, target_value_list = load_target(load_dir, case_num)
                    i = int(target[0])
                    j = int(target[1])
                    ce_mode = "positive" if target_value_list[0] >= 0 else "negative"

                    save_dir = get_save_dir(args, d, structure_kernel, case_num, run_num)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    output_file = get_output_file(args, save_dir, structure_kernel, case_num, run_num)

                    multilevel_model = Multilevel_Single_CE(
                        bge_model=bge_model,
                        data=X_train,
                        X=target_value_list,
                        i=i,
                        j=j,
                        save_dir=save_dir,
                        output_file=output_file,
                        max_outer_iter=args.max_outer_iter,
                        rng=rng,
                        structure_kernel=structure_kernel,
                        p_structure=p_structure_schedule(d, T=args.mcmc_iterations),
                        ce_mode=ce_mode,
                    )
                    probability_list = multilevel_model.calculate_probability(args.n, args.mcmc_iterations)

                    # ------------- Logging -------------
                    with open(output_file, "a") as f:
                        if args.dataset == "synthetic":
                            true_effects = np.linalg.inv(np.eye(d) - B)
                            log_and_print(f"case {case_num}, run {run_num}", f)
                            log_and_print(f"seed: {seed}", f)
                            log_and_print(f"B matrix: {B}", f)
                            log_and_print(f"True effects: {true_effects[i, j]}", f)
                        elif args.dataset == "sachs":
                            log_and_print(f"run {run_num}", f)
                            log_and_print(f"seed: {seed}", f)

                        for target_value, logp in zip(target_value_list, probability_list):
                            log_and_print(f"------------------", f)
                            log_and_print(f"Target value: {target_value}", f)
                            if ce_mode == "positive":
                                log_and_print(f"P(ACE > {target_value})={np.exp(logp)}, e^{logp}", f)
                            elif ce_mode == "negative":
                                log_and_print(f"P(ACE < {target_value})={np.exp(logp)}, e^{logp}", f)

                    run_results.append(
                        {
                            "run": run_num,
                            "seed": seed,
                            "case": case_num,
                            "target": [int(i), int(j)],
                            "ce_mode": ce_mode,
                            "target_values": [float(x) for x in target_value_list],
                            "log_probs": [float(x) for x in probability_list],
                            "probs": [float(np.exp(x)) for x in probability_list],
                        }
                    )

                results.append(run_results)

            results_path = base_dir / "all_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
