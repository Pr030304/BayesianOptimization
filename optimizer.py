from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .acquisitions import asy_ucb
from .domains import EuclideanDomain
from .euclidean_gp import EuclideanGPFitter, EuclideanGPFitterConfig
from .oper_utils import latin_hc_sampling
from .plotting import (
    plot_kernel_comparison_mean,
    plot_kernel_comparison_std,
    plot_regret_mean_with_error_bars,
    plot_regret_runs_with_mean,
    plot_regret_std,
    plot_simple_regret,
    plot_simple_regret_runs,
)
from .problem_loader import load_problem


@dataclass
class QueryInfo:
    step_idx: int
    point: np.ndarray
    stage: str
    curr_acq: str | None = None
    hp_tune_method: str | None = None
    send_time: float | None = None
    receive_time: float | None = None
    val: float | None = None
    true_val: float | None = None


@dataclass
class BOHistory:
    query_step_idxs: list[int] = field(default_factory=list)
    query_points: list[np.ndarray] = field(default_factory=list)
    query_vals: list[float] = field(default_factory=list)
    query_true_vals: list[float] = field(default_factory=list)
    query_qinfos: list[QueryInfo] = field(default_factory=list)
    query_acqs: list[str | None] = field(default_factory=list)
    query_hp_tune_methods: list[str | None] = field(default_factory=list)


@dataclass
class BOResult:
    best_point: np.ndarray
    best_value: float
    points: np.ndarray
    values: np.ndarray
    best_so_far_values: np.ndarray
    history: BOHistory | None = None
    simple_regret: np.ndarray | None = None
    simple_regret_std: np.ndarray | None = None
    regret_plot_path: str | None = None
    num_runs: int = 1
    best_points_runs: np.ndarray | None = None
    best_values_runs: np.ndarray | None = None
    simple_regret_runs: np.ndarray | None = None
    history_runs: list[BOHistory] | None = None
    extra_plot_paths: dict[str, str] | None = None


@dataclass
class KernelComparisonResult:
    results_by_kernel: dict[str, BOResult]
    plot_paths: dict[str, str]


class ScratchBO:
    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        domain_bounds,
        maximize: bool = True,
        num_init_evals: int | None = None,
        init_method: str = "latin_hc",
        init_capital: str | float | None = "default",
        init_capital_frac: float | None = None,
        kernel_type: str = "se",
        metric_warmup_evals: int = 20,
        metric_shrinkage: float = 0.1,
        metric_update_subset: str = "all",
        metric_subset_size: int | None = None,
        perturb_thresh: float = 1e-4,
        acq_opt_max_evals: int = -1,
        random_seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.objective = objective
        self.bounds = np.asarray(domain_bounds, dtype=float)
        self.domain = EuclideanDomain(self.bounds)
        self.maximize = maximize
        self.sign = 1.0 if maximize else -1.0
        self.dim = self.bounds.shape[0]
        self.num_init_evals = num_init_evals or max(4, 2 * self.dim)
        self.init_method = init_method
        self.init_capital = init_capital
        self.init_capital_frac = init_capital_frac
        self.kernel_type = kernel_type
        self.metric_warmup_evals = metric_warmup_evals
        self.metric_shrinkage = metric_shrinkage
        self.metric_update_subset = metric_update_subset
        self.metric_subset_size = metric_subset_size
        self.perturb_thresh = perturb_thresh
        self.acq_opt_max_evals = acq_opt_max_evals
        self.rng = np.random.default_rng(random_seed)
        self.verbose = verbose
        self.step_idx = 0
        self.gp = None
        self.metric_matrix: np.ndarray | None = None
        self.metric_source_gp = None
        self.curr_opt_val = -np.inf
        self.curr_opt_point: np.ndarray | None = None
        self.history = BOHistory()
        self.X: list[np.ndarray] = []
        self.Y_raw: list[float] = []
        self.Y: list[float] = []

    def _sample_uniform(self, n: int) -> np.ndarray:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        return self.rng.uniform(lows, highs, size=(n, self.dim))

    def _sample_initial_points(self, n: int) -> np.ndarray:
        if self.init_method == "rand":
            return self._sample_uniform(n)
        if self.init_method == "latin_hc":
            unit_samples = latin_hc_sampling(self.dim, n, rng=self.rng)
            lows = self.bounds[:, 0]
            highs = self.bounds[:, 1]
            return lows + unit_samples * (highs - lows)
        raise ValueError(f"Unsupported init_method: {self.init_method}")

    def _get_initial_eval_budget(self, total_budget: int) -> int:
        if self.init_capital == "default":
            init_capital = float(
                np.clip(
                    5 * self.dim,
                    max(5.0, 0.025 * total_budget),
                    max(5.0, 0.075 * total_budget),
                )
            )
        elif self.init_capital is not None:
            init_capital = float(self.init_capital)
        elif self.init_capital_frac is not None:
            init_capital = float(self.init_capital_frac) * float(total_budget)
        else:
            init_capital = None
        if init_capital is not None:
            init_evals = max(1, int(init_capital))
        else:
            init_evals = max(1, int(self.num_init_evals))
        return min(total_budget, init_evals)

    def _update_curr_optimum(self, point: np.ndarray, value: float) -> None:
        signed_value = self.sign * value
        if signed_value > self.curr_opt_val:
            self.curr_opt_val = signed_value
            self.curr_opt_point = np.asarray(point, dtype=float).copy()

    def _update_history(self, qinfo: QueryInfo) -> None:
        self.history.query_step_idxs.append(qinfo.step_idx)
        self.history.query_points.append(np.asarray(qinfo.point, dtype=float).copy())
        self.history.query_vals.append(float(qinfo.val))
        self.history.query_true_vals.append(float(qinfo.true_val))
        self.history.query_qinfos.append(qinfo)
        self.history.query_acqs.append(qinfo.curr_acq)
        self.history.query_hp_tune_methods.append(qinfo.hp_tune_method)

    def _dispatch_single_evaluation(self, qinfo: QueryInfo) -> float:
        point = np.asarray(qinfo.point, dtype=float)
        value = float(self.objective(point))
        qinfo.val = value
        qinfo.true_val = value
        qinfo.receive_time = time.time()
        self.X.append(point.copy())
        self.Y_raw.append(value)
        self.Y.append(self.sign * value)
        self._update_curr_optimum(point, value)
        self._update_history(qinfo)
        return value

    def _create_qinfo(self, point: np.ndarray, stage: str, curr_acq: str | None = None) -> QueryInfo:
        self.step_idx += 1
        return QueryInfo(
            step_idx=self.step_idx,
            point=np.asarray(point, dtype=float).copy(),
            stage=stage,
            curr_acq=curr_acq,
            hp_tune_method="ml-rand",
            send_time=time.time(),
        )

    def _log_evaluation(self, stage: str, x: np.ndarray, value: float) -> None:
        if not self.verbose:
            return
        point_str = np.asarray(x, dtype=float).tolist()
        print(f"{stage}: point={point_str}, value={value}")

    def _fit_gp(self, kernel_type: str, metric_matrix: np.ndarray | None = None):
        if kernel_type == "se":
            options = EuclideanGPFitterConfig(
                kernel_type="se",
                hp_tune_max_evals=-1,
                ml_hp_tune_opt="rand",
                use_same_bandwidth=False,
                mean_func_type="tune",
                noise_var_type="tune",
                rng=self.rng,
            )
        else:
            options = EuclideanGPFitterConfig(
                kernel_type=kernel_type,
                hp_tune_max_evals=max(500, 100 * self.dim),
                ml_hp_tune_opt="rand",
                use_same_bandwidth=False,
                mean_func_type="tune",
                noise_var_type="tune",
                metric_matrix=metric_matrix,
                rng=self.rng,
            )
        fitter = EuclideanGPFitter(list(self.X), np.asarray(self.Y), options=options)
        fit_type, gp, _ = fitter.fit_gp()
        if fit_type != "fitted_gp":
            raise RuntimeError(f"Unexpected Dragonfly GP fit type: {fit_type}")
        return gp

    def _get_metric_update_indices(self) -> np.ndarray:
        num_points = len(self.X)
        if num_points == 0:
            return np.asarray([], dtype=int)
        subset_size = self.metric_subset_size
        if subset_size is None or subset_size <= 0 or subset_size >= num_points:
            return np.arange(num_points, dtype=int)
        if self.metric_update_subset == "all":
            return np.arange(num_points, dtype=int)
        if self.metric_update_subset == "topk":
            return np.argsort(np.asarray(self.Y, dtype=float))[-subset_size:]
        if self.metric_update_subset == "recent":
            return np.arange(num_points - subset_size, num_points, dtype=int)
        raise ValueError(f"Unsupported metric_update_subset: {self.metric_update_subset}")

    def _compute_metric_matrix(self, gp) -> np.ndarray:
        metric_matrix = np.zeros((self.dim, self.dim), dtype=float)
        for idx in self._get_metric_update_indices():
            point = self.X[idx]
            gradient = gp.eval_mean_gradient(point)
            grad_norm = float(np.linalg.norm(gradient))
            gradient = gradient / (grad_norm + 1e-8)
            metric_matrix += np.outer(gradient, gradient)
        return metric_matrix

    def _normalize_metric_matrix(self, metric_matrix: np.ndarray) -> np.ndarray:
        trace = float(np.trace(metric_matrix))
        if trace <= 1e-12:
            return np.eye(self.dim)
        return metric_matrix / trace

    def _shrink_metric_matrix(self, metric_matrix: np.ndarray) -> np.ndarray:
        shrinkage = float(np.clip(self.metric_shrinkage, 0.0, 1.0))
        return (1.0 - shrinkage) * metric_matrix + shrinkage * np.eye(self.dim)

    def _update_metric_matrix(self) -> None:
        if self.kernel_type != "matrix_se" or self.metric_source_gp is None:
            return
        if len(self.X) < self.metric_warmup_evals:
            self.metric_source_gp = None
            self.metric_matrix = None
            return
        raw_metric_matrix = self._compute_metric_matrix(self.metric_source_gp)
        shrunk_metric_matrix = self._shrink_metric_matrix(raw_metric_matrix)
        self.metric_matrix = self._normalize_metric_matrix(shrunk_metric_matrix)
        self.metric_source_gp = None

    def _build_gp(self):
        if self.kernel_type == "se":
            return self._fit_gp("se")
        if self.kernel_type != "matrix_se":
            raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")

        active_metric_matrix = self.metric_matrix if self.metric_matrix is not None else np.eye(self.dim)
        gp = self._fit_gp("matrix_se", metric_matrix=active_metric_matrix)
        self.metric_source_gp = gp
        return gp

    def _main_loop_pre(self) -> None:
        self.gp = self._build_gp()

    def _get_acq_opt_max_evals(self, time_step: int) -> int:
        if isinstance(self.acq_opt_max_evals, int) and self.acq_opt_max_evals > 0:
            return int(self.acq_opt_max_evals)
        lead_const = 10 * min(5, self.dim) ** 2
        return int(np.clip(lead_const * np.sqrt(min(time_step, 1000)), 2000, 3e4))

    def _domain_diameter(self) -> float:
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        return float(np.linalg.norm(widths))

    def _is_too_close_to_existing_points(self, point: np.ndarray) -> bool:
        if not self.X:
            return False
        threshold = self.perturb_thresh * self._domain_diameter()
        if threshold <= 0:
            return False
        distances = [np.linalg.norm(point - prev_point) for prev_point in self.X]
        return min(distances) < threshold

    def _perturb_point_if_needed(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float).copy()
        if not self._is_too_close_to_existing_points(point):
            return point
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        for _ in range(25):
            perturbed = point + self.rng.normal(scale=0.01 * widths, size=self.dim)
            perturbed = np.clip(perturbed, self.bounds[:, 0], self.bounds[:, 1])
            if not self._is_too_close_to_existing_points(perturbed):
                return perturbed
        return self._sample_uniform(1)[0]

    def _get_next_ucb_point(self, gp) -> np.ndarray:
        anc_data = Namespace(
            max_evals=self._get_acq_opt_max_evals(max(self.step_idx, 1)),
            t=max(self.step_idx, 1),
            domain=self.domain,
            curr_max_val=self.curr_opt_val,
            eval_points_in_progress=[],
            rng=self.rng,
        )
        return np.asarray(asy_ucb(gp, anc_data), dtype=float)

    def _determine_next_query(self) -> QueryInfo:
        next_point = self._get_next_ucb_point(self.gp)
        next_point = self._perturb_point_if_needed(next_point)
        return self._create_qinfo(next_point, stage="bo", curr_acq="ucb")

    def _initialise(self, num_evals: int) -> None:
        num_init_evals = self._get_initial_eval_budget(num_evals)
        init_points = self._sample_initial_points(num_init_evals)
        for idx, point in enumerate(init_points, start=1):
            qinfo = self._create_qinfo(point, stage="init", curr_acq="init")
            value = self._dispatch_single_evaluation(qinfo)
            self._log_evaluation(f"init[{idx}]", point, value)

    def run(self, num_evals: int) -> BOResult:
        if not self.X:
            self._initialise(num_evals)
        while len(self.X) < num_evals:
            self._main_loop_pre()
            qinfo = self._determine_next_query()
            value = self._dispatch_single_evaluation(qinfo)
            self._update_metric_matrix()
            self._log_evaluation(f"bo[{len(self.X)}]", qinfo.point, value)
        values = np.asarray(self.Y_raw, dtype=float)
        if self.maximize:
            best_so_far_values = np.maximum.accumulate(values)
        else:
            best_so_far_values = np.minimum.accumulate(values)
        best_index = int(np.argmax(values) if self.maximize else np.argmin(values))
        return BOResult(
            best_point=np.asarray(self.X[best_index], dtype=float),
            best_value=float(values[best_index]),
            points=np.asarray(self.X, dtype=float),
            values=values,
            best_so_far_values=best_so_far_values,
            history=self.history,
        )


def _compute_simple_regret(best_so_far_values, true_opt_value: float, maximize: bool) -> np.ndarray:
    best_so_far_values = np.asarray(best_so_far_values, dtype=float)
    if maximize:
        regret = true_opt_value - best_so_far_values
    else:
        regret = best_so_far_values - true_opt_value
    return np.maximum(regret, 0.0)


def _default_regret_plot_path(problem_file: str, suffix: str) -> Path:
    problem_path = Path(problem_file).resolve()
    return problem_path.with_name(f"{problem_path.stem}_{suffix}.png")


def optimise_problem_file(
    problem_file: str,
    num_evals: int,
    random_seed: int | None = None,
    acq_opt_max_evals: int = -1,
    num_runs: int = 1,
    verbose: bool = True,
    kernel_override: str | None = None,
    regret_plot_path_override: str | Path | None = None,
) -> BOResult:
    problem = load_problem(problem_file)
    if kernel_override is not None:
        problem["kernel_type"] = kernel_override
    run_results: list[BOResult] = []
    start_time = time.time()
    for run_idx in range(num_runs):
        run_seed = None if random_seed is None else random_seed + run_idx
        run_start = time.time()
        optimiser = ScratchBO(
            objective=problem["objective"],
            domain_bounds=problem["domain_bounds"],
            maximize=problem["maximize"],
            num_init_evals=problem["num_init_evals"],
            init_method=problem["init_method"],
            init_capital=problem["init_capital"],
            init_capital_frac=problem["init_capital_frac"],
            kernel_type=problem["kernel_type"],
            metric_warmup_evals=problem["metric_warmup_evals"],
            metric_shrinkage=problem["metric_shrinkage"],
            metric_update_subset=problem["metric_update_subset"],
            metric_subset_size=problem["metric_subset_size"],
            perturb_thresh=problem["perturb_thresh"],
            acq_opt_max_evals=acq_opt_max_evals,
            random_seed=run_seed,
            verbose=verbose and num_runs == 1,
        )
        run_result = optimiser.run(num_evals=num_evals)
        run_results.append(run_result)

        if num_runs > 1:
            elapsed = time.time() - start_time
            run_time = time.time() - run_start
            avg_per_run = elapsed / (run_idx + 1)
            remaining = avg_per_run * (num_runs - (run_idx + 1))
            best_point = run_result.best_point.tolist()
            best_value = run_result.best_value
            print(
                f"run {run_idx + 1}/{num_runs} done in {run_time:.1f}s | "
                f"best_point={best_point} best_value={best_value} | "
                f"ETA ~ {remaining/60:.1f} min"
            )

    result = run_results[0]
    result.num_runs = num_runs
    result.best_points_runs = np.asarray([run.best_point for run in run_results], dtype=float)
    result.best_values_runs = np.asarray([run.best_value for run in run_results], dtype=float)
    result.history_runs = [run.history for run in run_results]

    true_opt_value = problem["true_opt_value"]
    if true_opt_value is not None:
        result.simple_regret_runs = np.asarray(
            [
                _compute_simple_regret(
                    run.best_so_far_values,
                    true_opt_value=true_opt_value,
                    maximize=problem["maximize"],
                )
                for run in run_results
            ],
            dtype=float,
        )
        result.simple_regret = result.simple_regret_runs.mean(axis=0)
        result.simple_regret_std = result.simple_regret_runs.std(axis=0)
        regret_plot_path = regret_plot_path_override if regret_plot_path_override is not None else problem["regret_plot_path"]
        if regret_plot_path is None:
            regret_plot_path = _default_regret_plot_path(problem_file, "simple_regret")
        if num_runs == 1:
            result.regret_plot_path = plot_simple_regret(result.simple_regret, regret_plot_path)
        else:
            result.regret_plot_path = plot_regret_mean_with_error_bars(
                result.simple_regret_runs,
                regret_plot_path,
            )
            result.extra_plot_paths = {
                "mean_error_bars": result.regret_plot_path,
                "std": plot_regret_std(
                    result.simple_regret_runs,
                    _default_regret_plot_path(problem_file, f"{problem['kernel_type']}_std"),
                    title=f"Std Deviation of Simple Regret ({problem['kernel_type']})",
                ),
                "runs_mean": plot_regret_runs_with_mean(
                    result.simple_regret_runs,
                    _default_regret_plot_path(problem_file, f"{problem['kernel_type']}_runs_mean"),
                    title=f"Simple Regret Across Runs ({problem['kernel_type']})",
                ),
            }

    if num_runs > 1:
        best_run_idx = int(np.argmax(result.best_values_runs) if problem["maximize"] else np.argmin(result.best_values_runs))
        result.best_point = result.best_points_runs[best_run_idx]
        result.best_value = float(result.best_values_runs[best_run_idx])

    return result


def optimise_problem_file_compare_kernels(
    problem_file: str,
    num_evals: int,
    random_seed: int | None = None,
    acq_opt_max_evals: int = -1,
    num_runs: int = 1,
    verbose: bool = True,
) -> KernelComparisonResult:
    results_by_kernel: dict[str, BOResult] = {}
    for kernel_name in ["se", "matrix_se"]:
        result = optimise_problem_file(
            problem_file=problem_file,
            num_evals=num_evals,
            random_seed=random_seed,
            acq_opt_max_evals=acq_opt_max_evals,
            num_runs=num_runs,
            verbose=verbose,
            kernel_override=kernel_name,
            regret_plot_path_override=_default_regret_plot_path(problem_file, f"{kernel_name}_mean"),
        )
        results_by_kernel[kernel_name] = result

    regret_runs_by_kernel = {
        kernel_name: result.simple_regret_runs
        for kernel_name, result in results_by_kernel.items()
        if result.simple_regret_runs is not None
    }
    plot_paths: dict[str, str] = {}
    if len(regret_runs_by_kernel) == 2:
        plot_paths["combined_mean"] = plot_kernel_comparison_mean(
            regret_runs_by_kernel,
            _default_regret_plot_path(problem_file, "se_vs_matrix_se_mean"),
        )
        plot_paths["combined_std"] = plot_kernel_comparison_std(
            regret_runs_by_kernel,
            _default_regret_plot_path(problem_file, "se_vs_matrix_se_std"),
        )
        for kernel_name, result in results_by_kernel.items():
            if result.extra_plot_paths is not None:
                plot_paths[f"{kernel_name}_mean"] = result.extra_plot_paths["mean_error_bars"]
                plot_paths[f"{kernel_name}_std"] = result.extra_plot_paths["std"]
                plot_paths[f"{kernel_name}_runs_mean"] = result.extra_plot_paths["runs_mean"]

    return KernelComparisonResult(results_by_kernel=results_by_kernel, plot_paths=plot_paths)
