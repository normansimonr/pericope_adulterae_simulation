import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import yaml

from pasim.execution.runner import run_single


def _run_single_unpacked(args_tuple):
    """Helper function to unpack arguments for run_single in a map context."""
    params_path, seed_val = args_tuple
    return run_single(params_path, seed_val)


def run_parallel(params_path: str, base_seed: Optional[int] = None) -> None:
    """
    Orchestrates multiple independent simulation runs of the same experiment in parallel.

    This function is designed for Monte Carlo replication, launching N independent
    simulations using the existing `run_single()` function.

    Args:
        params_path: The file path to the YAML configuration file for the experiment.
                     This file must contain an 'n_runs' field specifying the number
                     of parallel runs.
        base_seed: An optional base seed for the random number generators. If provided,
                   each parallel run will derive its seed from this base seed. If not
                   provided, a default base seed (e.g., 42) will be used to ensure
                   reproducibility of the parallel execution batch.

    Raises:
        FileNotFoundError: If the specified `params_path` does not exist.
        ValueError: If 'n_runs' is not specified in the params file or is invalid.
    """
    params_file_path = Path(params_path)
    if not params_file_path.is_file():
        raise FileNotFoundError(f"Parameter file not found at: {params_path}")

    with open(params_file_path, "r") as f:
        params = yaml.safe_load(f)

    n_runs = params.get("n_runs")
    if n_runs is None or not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError(f"The params file '{params_path}' must contain a positive integer field 'n_runs'.")

    # Determine the base seed for deriving individual run seeds
    if base_seed is None:
        # If no base_seed is provided to run_parallel, check params for a seed.
        # If params has 'seed', use it as base. Otherwise, use a hardcoded default.
        base_seed = params.get("seed", 42)  # Default to 42 if no seed in params or function call

    # Generate unique seeds for each parallel run
    # Each run will get a seed derived from the base_seed to ensure independent RNG streams
    # and reproducibility of the entire batch if the base_seed is fixed.
    seeds = [base_seed + i for i in range(n_runs)]

    # Determine the number of worker processes
    num_workers = os.cpu_count()
    if num_workers is None or num_workers < 1:
        num_workers = 1  # Fallback to 1 worker if cpu_count is not available or invalid

    print(f"Launching {n_runs} simulation runs in parallel using {num_workers} processes...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare arguments for each run_single call
        # executor.map takes a function and then iterables of arguments
        # It applies func(arg1[i], arg2[i], ...) for each i
        args = [(params_path, seed_val) for seed_val in seeds]

        # Use starmap to pass arguments from the list of tuples
        # It blocks until all tasks are completed.
        # No results are returned as `run_single` persists output directly.
        for _ in executor.map(_run_single_unpacked, args):
            pass  # Consume the iterator to ensure all futures are processed


# Update src/pasim/execution/__init__.py to expose run_parallel
# I will make sure to include this change in the next commit.
