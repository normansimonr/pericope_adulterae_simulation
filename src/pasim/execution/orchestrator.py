from pathlib import Path
from typing import Any, Dict, Union

import yaml

from pasim.execution.parallel import run_parallel


def run_experiment(params_path: Union[Path, str]) -> Dict[str, Any]:
    """
    Executes a complete experiment definition, orchestrating multiple parallel
    Monte Carlo runs based on the provided parameters file.

    This function serves as the high-level entrypoint for running experiments,
    leveraging the parallel execution and retry logic.

    Args:
        params_path: The file path to the YAML configuration file defining the experiment.
                     This file must contain 'n_runs' and may optionally contain 'max_retries'
                     and a base 'seed'.

    Returns:
        A dictionary summarizing the overall experiment execution, including counts
        of successful and failed runs, and details of any failures.
    """
    params_file_path = Path(params_path)
    if not params_file_path.is_file():
        raise FileNotFoundError(f"Experiment params file not found at: {params_file_path}")

    with open(params_file_path, "r") as f:
        params = yaml.safe_load(f)

    # Extract n_runs and max_retries, seed from the params file
    n_runs = params.get("n_runs")
    if n_runs is None or not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError(f"Experiment params file '{params_file_path}' must contain a positive integer field 'n_runs'.")

    # max_retries is optional, run_parallel will handle its default
    # base_seed is optional, run_parallel will handle its default
    base_seed = params.get("seed")

    # Invoke the parallel orchestrator
    experiment_summary = run_parallel(str(params_file_path), base_seed=base_seed)

    print("\nExperiment Summary:")
    print(f"Total Runs: {experiment_summary['total_runs']}")
    print(f"Successful Runs: {experiment_summary['successful_runs']}")
    print(f"Failed Runs: {experiment_summary['failed_runs']}")
    if experiment_summary["failed_runs"] > 0:
        print(f"Failure Details: {experiment_summary['failure_records']}")

    return experiment_summary
