# Bayesian Optimization with SE and Learned Geometry Kernels

This repository contains a lightweight Bayesian Optimization (BO) implementation for Euclidean box-constrained problems. The code uses Gaussian Process (GP) surrogate models together with an Upper Confidence Bound (UCB) acquisition function, and supports comparison between a standard Squared Exponential (SE) kernel and a learned-geometry `Matrix-SE` kernel.

The project is intended for studying how geometry-aware kernels affect optimization quality, regret, stability, and runtime across repeated runs.

## Overview

Bayesian Optimization is useful for optimizing functions that are:

- expensive to evaluate,
- non-convex,
- gradient-free,
- or treated as black-box objectives.

Instead of evaluating the objective everywhere, BO fits a GP surrogate to previously observed points and then uses an acquisition function to decide where to sample next.

This codebase supports:

- standard SE-kernel BO,
- learned-geometry Matrix-SE BO,
- repeated runs with shared experiment settings,
- simple regret computation,
- regret plots and kernel-comparison plots,
- configurable acquisition-optimization budgets.

## Repository Structure

```text
.
├── __init__.py
├── acquisitions.py
├── cli.py
├── domains.py
├── euclidean_gp.py
├── example_problem.py
├── gp_core.py
├── kernel.py
├── oper_utils.py
├── optimizer.py
├── plotting.py
├── problem_loader.py
└── utils.py
```

## Main Files

- `cli.py`
  Command-line entrypoint for running optimization experiments.

- `optimizer.py`
  Main BO loop, repeated-run logic, regret computation, and SE vs Matrix-SE comparison.

- `euclidean_gp.py`
  GP fitting utilities for SE and Matrix-SE kernels.

- `kernel.py`
  Kernel definitions used by the GP model.

- `acquisitions.py`
  Acquisition-function helpers.

- `problem_loader.py`
  Loads user-defined benchmark/problem files.

- `plotting.py`
  Utilities for plotting simple regret, kernel comparisons, and error bars.

- `example_problem.py`
  Example problem file showing the expected interface.

## Core Idea

The main comparison in this repository is between:

### 1. SE Kernel

A standard squared exponential / RBF kernel.

### 2. Matrix-SE Kernel

A geometry-aware extension that uses a learned matrix `M` to define distances inside the kernel. This allows the model to adapt to rotated or directionally anisotropic structure in the objective.

The learned geometry matrix is built from normalized surrogate gradients and stabilized using:

- shrinkage regularization,
- trace normalization,
- delayed updates,
- subset-based updates such as focusing on high-value points.

## Requirements

This code requires:

- Python 3.10+
- `numpy`
- `scipy`
- `matplotlib`

Install dependencies with:

```bash
pip install numpy scipy matplotlib
```

If you are using an existing virtual environment:

```bash
source dfly/bin/activate
```

## Problem File Format

The optimizer expects a Python file defining at least:

- `domain_bounds`
- `objective(x)`

Optional fields include:

- `maximize`
- `num_init_evals`
- `init_method`
- `init_capital`
- `init_capital_frac`
- `kernel_type`
- `metric_warmup_evals`
- `metric_update_subset`
- `metric_subset_size`
- `metric_shrinkage`
- `true_opt_value`
- `regret_plot_path`

### Minimal Example

```python
import numpy as np

domain_bounds = [[0.0, 1.0], [0.0, 1.0]]
maximize = True

def objective(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(-(x[0] - 0.5) ** 2 - (x[1] - 0.25) ** 2)
```

## How to Run

### 1. Single optimization run

```bash
python -m cli --problem example_problem.py --num-evals 100 --seed 0
```

If you are running from the parent directory and treating this as a package, use:

```bash
python -m scratch.cli --problem scratch/example_problem.py --num-evals 100 --seed 0
```

### 2. Multiple independent runs

```bash
python -m cli --problem example_problem.py --num-evals 100 --num-runs 5 --seed 0
```

This reports:

- best value per run,
- best overall point,
- best overall value,
- final mean simple regret,
- final standard deviation of simple regret.

### 3. Compare SE vs Matrix-SE

```bash
python -m cli \
  --problem example_problem.py \
  --num-evals 100 \
  --num-runs 5 \
  --seed 0 \
  --acq-opt-max-evals 5000 \
  --compare-kernels
```

This runs both:

- standard SE BO,
- learned-geometry Matrix-SE BO,

under the same setup and produces comparison statistics and plots.

## CLI Arguments

### `--problem`

Path to the Python file defining the optimization problem.

### `--num-evals`

Total BO evaluation budget.

### `--num-runs`

Number of independent repeated runs.

### `--seed`

Random seed for reproducibility.

### `--compare-kernels`

Runs both `se` and `matrix_se` and compares them.

### `--acq-opt-max-evals`

Budget for acquisition optimization.

- If positive, a fixed budget is used.
- If set to `-1`, the code uses a Dragonfly-style adaptive schedule that increases with iteration count.

## Evaluation Metric

The main evaluation metric is simple regret:

\[
r_t = f(x^\ast) - \max_{i \le t} f(x_i)
\]

where:

- `x*` is the true optimum,
- `x_i` are sampled points up to iteration `t`.

Lower simple regret indicates better optimization performance.

The code can compute:

- regret per run,
- mean regret across runs,
- regret standard deviation,
- error-bar plots.

## Geometry Learning in Matrix-SE

For the learned-geometry kernel, the matrix `M` is updated from GP mean gradients evaluated at selected previously sampled points.

To stabilize learning, the code uses:

### Gradient normalization

Each gradient is normalized before contributing to the geometry update.

### Geometry aggregation

The geometry matrix is formed from outer products of normalized gradients.

### Shrinkage regularization

\[
M \leftarrow (1-\lambda)M + \lambda I
\]

### Trace normalization

\[
M \leftarrow \frac{M}{\mathrm{tr}(M)}
\]

### Delayed updates

Geometry learning is activated only after a warmup phase.

### Subset-based updates

The update can use:

- all sampled points,
- recent points,
- or best-performing points.

## Output

Depending on the mode, the code prints:

- best point,
- best value,
- best value per run,
- mean simple regret,
- standard deviation of regret,
- plot file paths.

Generated plots may include:

- mean simple regret curves,
- regret standard deviation curves,
- SE vs Matrix-SE comparison plots,
- run-wise regret summaries.

## Notes

- Generated `.png` plots and `.pdf` files are intentionally excluded from version control through `.gitignore`.
- Runtime can become large when `--acq-opt-max-evals` is high, especially for repeated runs and kernel comparison mode.
- The implementation is focused on Euclidean box domains.

## Example Use Cases

This repository is useful for:

- learning Bayesian Optimization from scratch,
- comparing kernel behavior,
- studying learned geometry in GP kernels,
- running BO benchmark experiments,
- visualizing regret across repeated runs.

## Future Extensions

Possible extensions include:

- additional acquisition functions such as EI or Thompson Sampling,
- constrained BO,
- higher-dimensional benchmark studies,
- multi-fidelity BO,
- improved learned-geometry update rules.

## Author

Pranjali Singh
