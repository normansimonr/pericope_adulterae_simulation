import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from pasim.core.rng import RNGContext
from pasim.execution.runner import run_single

logger = logging.getLogger(__name__)


def _run_wrapper(
    params_path: str,
    seed: int,
    max_retries: int,
) -> Dict[str, Any]:
    """
    Wrapper function to execute a single simulation run with retry logic.
    Returns a dictionary indicating success/failure and relevant data.
    """
    for attempt in range(max_retries + 1):
        try:
            result = run_single(params_path=params_path, seed=seed)
            return {
                "status": "success",
                "result": result,
                "seed": seed,
                "retries": attempt,
                "error": None,
            }
        except Exception as e:
            logger.warning(f"Run (seed: {seed}) failed on attempt {attempt}/{max_retries}: {e}")
            if attempt >= max_retries:
                logger.error(f"Run (seed: {seed}) failed after {max_retries} retries.")
                return {
                    "status": "failure",
                    "result": None,
                    "seed": seed,
                    "retries": attempt,
                    "error": str(e),
                }
    # This line should ideally not be reached
    return {
        "status": "failure",
        "result": None,
        "seed": seed,
        "retries": max_retries,
        "error": "Unknown error in _run_wrapper",
    }


def run_parallel(
    params_path: str,
    n_runs: int,
    max_retries: int,
    seed: Optional[int] = None,
    num_processes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Orchestrates multiple independent simulation runs in parallel.

    Args:
        params_path (str): Path to the YAML configuration file for the experiment.
        n_runs (int): The number of independent simulation runs to execute.
        max_retries (int): The maximum number of times to retry a failed run.
        seed (Optional[int]): An optional master seed for the random number generator.
                              If None, a default fixed seed is used.
        num_processes (Optional[int]): The number of parallel processes to use.
                                       If None, it defaults to the number of CPUs.

    Returns:
        Dict[str, Any]: A summary of the parallel execution, including counts of
                        successful/failed runs and detailed failure records.
    """
    if n_runs <= 0:
        return {
            "successful_runs": 0,
            "failed_runs": 0,
            "failure_records": [],
            "total_runs_attempted": 0,
        }

    rng_context = RNGContext(seed=seed)
    # Generate a unique seed for each run
    run_seeds = [gen.integers(0, 2**32 - 1) for gen in rng_context.spawn(n_runs)]

    successful_runs_count = 0
    failed_runs_count = 0
    failure_records: List[Dict[str, Any]] = []

    # Use default number of processes if not specified
    if num_processes is None:
        num_processes = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all runs to the executor
        futures = {executor.submit(_run_wrapper, params_path, run_seeds[i], max_retries): run_seeds[i] for i in range(n_runs)}

        for future in as_completed(futures):
            current_seed = int(futures[future])  # Retrieve and convert to int
            try:
                result_data = future.result()
                if result_data["status"] == "success":
                    successful_runs_count += 1
                    logger.info(f"Run (seed: {current_seed}) completed successfully after {result_data['retries']} retries.")
                else:
                    failed_runs_count += 1
                    failure_records.append({
                        "seed": current_seed,
                        "error": result_data["error"],
                        "attempt": result_data["retries"],
                    })
                    logger.error(f"Run (seed: {current_seed}) failed after {result_data['retries']} retries: {result_data['error']}")
            except Exception as exc:
                failed_runs_count += 1
                failure_records.append({
                    "seed": current_seed,
                    "error": f"Unhandled exception: {str(exc)}",
                    "attempt": 0,  # Unhandled exception means no retries by _run_wrapper
                })
                logger.critical(f"Run (seed: {current_seed}) generated an unhandled exception: {exc}")

    return {
        "successful_runs": successful_runs_count,
        "failed_runs": failed_runs_count,
        "failure_records": failure_records,
        "total_runs_attempted": n_runs,
    }
