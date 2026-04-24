from __future__ import annotations

import argparse

from .optimizer import optimise_problem_file, optimise_problem_file_compare_kernels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential BO with SE GP and random-search UCB.")
    parser.add_argument("--problem", required=True, help="Path to a python file with objective/domain_bounds.")
    parser.add_argument("--num-evals", required=True, type=int, help="Total number of BO evaluations.")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of independent BO runs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--compare-kernels",
        action="store_true",
        help="Run both SE and Matrix-SE sequentially and generate combined comparison plots.",
    )
    parser.add_argument(
        "--acq-opt-max-evals",
        type=int,
        default=-1,
        help="Random-search budget for acquisition optimization. Use -1 for Dragonfly-style time-dependent scheduling.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.compare_kernels:
        comparison = optimise_problem_file_compare_kernels(
            problem_file=args.problem,
            num_evals=args.num_evals,
            num_runs=args.num_runs,
            random_seed=args.seed,
            acq_opt_max_evals=args.acq_opt_max_evals,
        )
        for kernel_name, result in comparison.results_by_kernel.items():
            print(f"{kernel_name}_best_point:", result.best_point.tolist())
            print(f"{kernel_name}_best_value:", result.best_value)
            if result.simple_regret is not None:
                print(f"{kernel_name}_final_mean_simple_regret:", float(result.simple_regret[-1]))
            if result.simple_regret_std is not None:
                print(f"{kernel_name}_final_std_simple_regret:", float(result.simple_regret_std[-1]))
        for plot_name, plot_path in comparison.plot_paths.items():
            print(f"{plot_name}_plot:", plot_path)
        return

    result = optimise_problem_file(
        problem_file=args.problem,
        num_evals=args.num_evals,
        num_runs=args.num_runs,
        random_seed=args.seed,
        acq_opt_max_evals=args.acq_opt_max_evals,
    )
    if result.num_runs == 1:
        print("best_point:", result.best_point.tolist())
        print("best_value:", result.best_value)
    else:
        print("num_runs:", result.num_runs)
        print("best_value_per_run:", result.best_values_runs.tolist())
        print("best_overall_point:", result.best_point.tolist())
        print("best_overall_value:", result.best_value)
        if result.simple_regret is not None:
            print("final_mean_simple_regret:", float(result.simple_regret[-1]))
        if result.simple_regret_std is not None:
            print("final_std_simple_regret:", float(result.simple_regret_std[-1]))
    if result.regret_plot_path is not None:
        print("simple_regret_plot:", result.regret_plot_path)
    if result.extra_plot_paths is not None:
        for plot_name, plot_path in result.extra_plot_paths.items():
            print(f"{plot_name}_plot:", plot_path)


if __name__ == "__main__":
    main()
