import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from pasim.execution.runner import run_single


def _run_single_with_retry(run_index: int, params_path: str, seed: int, max_retries: int) -> Dict[str, Any]:
    """
    Worker function that executes a single run with a retry mechanism.

    This function is designed to be called in a parallel worker process. It attempts
    to execute `run_single` and retries on failure up to `max_retries`.

    Args:
        run_index: The index of the run (for tracking purposes).
        params_path: The file path to the experiment's parameters file.
        seed: The seed for the random number generator.
        max_retries: The maximum number of retry attempts.

    Returns:
        A dictionary indicating the status of the run ('success' or 'failed'),
        the run index, and any recorded failures.
    """
    failures = []
    for attempt in range(max_retries + 1):
        try:
            run_single(params_path, seed)
            return {"status": "success", "run_index": run_index, "failures": []}
        except Exception as e:
            failures.append({
                "run_index": run_index,
                "attempt": attempt + 1,
                "exception": repr(e),
            })
            # Small backoff before retrying
            time.sleep(0.1)

    # If all attempts fail
    return {"status": "failed", "run_index": run_index, "failures": failures}


def run_parallel(params_path: str, base_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Orchestrates multiple independent simulation runs with retries.

    This function launches N independent simulations in parallel, handling failures
    and retries for each run.

    Args:
        params_path: The file path to the YAML configuration file for the experiment.
                     Must contain 'n_runs' and optionally 'max_retries'.
        base_seed: An optional base seed for the random number generators.

    Returns:
        A dictionary summarizing the execution results.
    """
    params_file_path = Path(params_path)
    if not params_file_path.is_file():
        raise FileNotFoundError(f"Parameter file not found at: {params_path}")

    with open(params_file_path, "r") as f:
        params = yaml.safe_load(f)

    n_runs = params.get("n_runs")
    if n_runs is None or not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError(f"The params file '{params_path}' must contain a positive integer field 'n_runs'.")

    max_retries = params.get("max_retries", 1)  # Default to 1 retry

    if base_seed is None:
        base_seed = params.get("seed", 42)

    seeds = [base_seed + i for i in range(n_runs)]

    num_workers = os.cpu_count() or 1

    print(f"Launching {n_runs} simulation runs in parallel using {num_workers} processes...")

    args_list = [(i, params_path, seeds[i], max_retries) for i in range(n_runs)]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use map to submit all tasks and collect results
        # The helper function handles the retry logic internally.
        for result in executor.map(_run_single_with_retry, *zip(*args_list)):
            results.append(result)

    successful_runs = sum(1 for r in results if r["status"] == "success")
    failed_runs = n_runs - successful_runs
    failure_records = [r["failures"] for r in results if r["status"] == "failed"]
    # Flatten the list of lists of failures
    failure_records = [item for sublist in failure_records for item in sublist]

    summary = {
        "total_runs": n_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "failure_records": failure_records,
    }

    print("Parallel execution complete.")
    return summary
