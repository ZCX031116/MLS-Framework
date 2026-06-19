import argparse
import os
import json
import random
import numpy as np
from pathlib import Path
from src.bge import BGe
from mls_frame_multi_ce import Multilevel_Multi_CE
from configs.multi_ce_constraints import get_multi_ce_constraints
from src.helper_func import (
    pairwise_linear_ce_no_params,
    p_structure_schedule,
    log_and_print,
)

def _csv_ints(value):
    return [int(x) for x in value.split(",") if x.strip()]


def _csv_floats(value):
    return [float(x) for x in value.split(",") if x.strip()]


def _csv_strings(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi causal-effect MLS experiments.")
    parser.add_argument("--dataset", choices=["sachs", "synthetic"], default="sachs")
    parser.add_argument("--d-list", type=_csv_ints, default=[11], help="Comma-separated graph sizes.")
    parser.add_argument("--cases", type=_csv_ints, default=[1], help="Comma-separated synthetic case numbers.")
    parser.add_argument("--kernels", type=_csv_strings, default=["PARNI"], help="Comma-separated structure kernels.")
    parser.add_argument(
        "--constraint-names",
        type=_csv_strings,
        default=None,
        help="Optional comma-separated constraint set names to run, e.g. weak,strong.",
    )
    parser.add_argument("--x-levels", type=_csv_floats, default=[0.0], help="Comma-separated score levels.")
    parser.add_argument("--n", type=int, default=200, help="Number of MLS samples per level.")
    parser.add_argument("--mcmc-iterations", type=int, default=2000)
    parser.add_argument("--run-num", type=int, default=1, help="Number of repeated runs.")
    parser.add_argument("--max-outer-iter", type=int, default=10)
    parser.add_argument("--alpha-u", type=float, default=10)
    parser.add_argument("--params-per-graph", type=int, default=500)
    parser.add_argument("--mean", type=int, default=0, help="Synthetic data mean directory.")
    parser.add_argument("--train-size", type=int, default=1000, help="Synthetic train size used in data filenames.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for generating per-run RNG seeds.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def load_data(args, d, case_num):
    if args.dataset == "synthetic":
        load_case_dir = args.data_dir / f"mean={args.mean}" / f"d={d}" / f"case{case_num}"
        G = np.load(load_case_dir / f"G_{d}Nodes_train_size_{args.train_size}.npy")
        B = np.load(load_case_dir / f"B_{d}Nodes_train_size_{args.train_size}.npy")
        X_train = np.load(load_case_dir / f"train_{d}Nodes_train_size_{args.train_size}.npy")
        return G, B, X_train

    if args.dataset == "sachs":
        load_dir = args.data_dir / "sachs"
        G = np.load(load_dir / "sachs_graph.npy")
        B = empty_array = np.zeros_like(G)  # Not used for Sachs, but keeping the return structure consistent
        X_train = np.load(load_dir / "sachs_data.npy")
        return G, B, X_train

    raise ValueError(f"Unknown dataset={args.dataset!r}")


def get_save_dir(args, d, constraint_name, structure_kernel, case_num, run_num):
    if args.dataset == "synthetic":
        return (
            args.results_dir
            / structure_kernel
            / f"multi-CE_{constraint_name}"
            / f"d={d}"
            / f"case{case_num}"
            / f"run_{run_num}"
        )

    return args.results_dir / "multi-CE_sachs" / structure_kernel / f"run_{run_num}"


def main():
    args = parse_args()
    seed_source = random.Random(args.seed)
    # d = 4,8,16,32 for synthetic dataset, d = 11 for sachs dataset
    for d in args.d_list:
        ce_constraints_list = get_multi_ce_constraints(d)
        if args.constraint_names is not None:
            wanted_constraint_names = set(args.constraint_names)
            ce_constraints_list = [
                item for item in ce_constraints_list
                if item[0] in wanted_constraint_names
            ]
            if not ce_constraints_list:
                raise ValueError(f"No matching constraint sets for d={d}: {sorted(wanted_constraint_names)}")

        for name, ce_constraints in ce_constraints_list:
            # structure_kernel = "PARNI" / "Structure_MCMC"
            for structure_kernel in args.kernels:
                for num in args.cases:
                    results = []
                    for run_num in range(1, args.run_num + 1):
                        seed = seed_source.randint(0, 10000)
                        rng = np.random.default_rng(seed)
                        bge_model = BGe(d=d, alpha_u=args.alpha_u)
                        save_dir = get_save_dir(args, d, name, structure_kernel, num, run_num)
                        G, B, X_train = load_data(args, d, num)

                        ce = pairwise_linear_ce_no_params(
                            np.copy([G]),
                            X_train,
                            bge_model,
                            params_per_graph=args.params_per_graph,
                            avg=True,
                            return_B=False,
                        )

                        for (src, end, op, thr, sc) in ce_constraints:
                            print(f"approx CE({src}->{end}) = {ce[int(src), int(end)]:.6g}  constraint: CE {op} {thr}")

                        tmp = Path(save_dir)
                        tmp.mkdir(parents=True, exist_ok=True)
                        output_file = os.path.join(tmp, f"{structure_kernel}_run{run_num}_output_results.txt")

                        multilevel_model = Multilevel_Multi_CE(
                            bge_model=bge_model,
                            data=X_train,
                            X=args.x_levels,
                            ce_constraints=ce_constraints,
                            save_dir=tmp,
                            output_file=output_file,
                            max_outer_iter=args.max_outer_iter,
                            rng=rng,
                            structure_kernel=structure_kernel,
                            p_structure=p_structure_schedule(d, T=args.mcmc_iterations),
                        )

                        probability_list = multilevel_model.calculate_probability(args.n, args.mcmc_iterations)

                        # ------------- Logging -------------
                        with open(output_file, "a") as f:
                            log_and_print(f"run {run_num}", f)
                            log_and_print(f"seed: {seed}", f)
                            log_and_print(f"constraints: {ce_constraints}", f)
                            for (src, end, op, thr, sc) in ce_constraints:
                                log_and_print(f"approx CE({src}->{end}) = {ce[int(src), int(end)]:.6g}", f)

                            for L, logp in zip(args.x_levels, probability_list):
                                log_and_print(f"------------------", f)
                                log_and_print(f"Score level L: {float(L)}", f)
                                log_and_print(f"Condition: {multilevel_model._condition_str(L)}", f)
                                log_and_print(f"P(score >= {float(L)}) = {np.exp(logp)}, e^{logp}", f)

                        results.append({
                            "run": run_num,
                            "seed": seed,
                            "case": num,
                            "X_levels": [float(x) for x in args.x_levels],
                            "log_probs": [float(x) for x in probability_list],
                            "probs": [float(np.exp(x)) for x in probability_list],
                        })

                    results_path = tmp.parent / "all_results.json"
                    results_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
