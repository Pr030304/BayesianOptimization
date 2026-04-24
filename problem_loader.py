from __future__ import annotations

import importlib.util
from pathlib import Path


def load_problem_module(problem_file: str | Path):
    problem_path = Path(problem_file).resolve()
    spec = importlib.util.spec_from_file_location(problem_path.stem, problem_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load problem file: {problem_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_problem(problem_file: str | Path):
    module = load_problem_module(problem_file)
    if not hasattr(module, "objective"):
        raise AttributeError(f"{problem_file} must define an `objective(x)` function.")
    if not hasattr(module, "domain_bounds"):
        raise AttributeError(f"{problem_file} must define `domain_bounds`.")
    maximize = getattr(module, "maximize", True)
    num_init_evals = getattr(module, "num_init_evals", None)
    init_method = getattr(module, "init_method", "latin_hc")
    init_capital = getattr(module, "init_capital", "default")
    init_capital_frac = getattr(module, "init_capital_frac", None)
    kernel_type = getattr(module, "kernel_type", "se")
    metric_warmup_evals = getattr(module, "metric_warmup_evals", 20)
    metric_shrinkage = getattr(module, "metric_shrinkage", 0.1)
    metric_update_subset = getattr(module, "metric_update_subset", "all")
    metric_subset_size = getattr(module, "metric_subset_size", None)
    perturb_thresh = getattr(module, "perturb_thresh", 1e-4)
    true_opt_value = getattr(module, "true_opt_value", None)
    regret_plot_path = getattr(module, "regret_plot_path", None)
    return {
        "objective": module.objective,
        "domain_bounds": module.domain_bounds,
        "maximize": maximize,
        "num_init_evals": num_init_evals,
        "init_method": init_method,
        "init_capital": init_capital,
        "init_capital_frac": init_capital_frac,
        "kernel_type": kernel_type,
        "metric_warmup_evals": metric_warmup_evals,
        "metric_shrinkage": metric_shrinkage,
        "metric_update_subset": metric_update_subset,
        "metric_subset_size": metric_subset_size,
        "perturb_thresh": perturb_thresh,
        "true_opt_value": true_opt_value,
        "regret_plot_path": regret_plot_path,
    }
