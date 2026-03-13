import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from pasim.execution.run_plan import RunSpec, generate_run_plan
from pasim.execution.runner import run_single

logger = logging.getLogger(__name__)


def _run_wrapper(
    params_path: str,
    run_spec: RunSpec,
    regime: Optional[str],
    max_retries: int,
    persistence_level: str = "full",
) -> Dict[str, Any]:
    """
    Wrapper function to execute a single simulation run for one or more regimes with retry logic.
    Returns a dictionary indicating success/failure and relevant data.
    """
    for attempt in range(max_retries + 1):
        try:
            # We explicitly discard the full in-memory result object here
            # to prevent it from being pickled and sent back to the main process,
            # which can cause massive memory spikes for large graphs.
            # All critical data is already persisted to disk.
            run_single(
                params_path=params_path,
                seed=run_spec.seed,
                persistence_level=persistence_level,
                run_id=run_spec.run_id,
                regime=regime,
            )
            return {
                "status": "success",
                "run_id": run_spec.run_id,
                "regime": regime,
                "seed": run_spec.seed,
                "retries": attempt,
                "error": None,
            }
        except Exception as e:
            logger.warning(
                f"Run {run_spec.run_id} ({regime or 'both'}, seed: {run_spec.seed}) failed on attempt {attempt}/{max_retries}: {e}"
            )
            if attempt >= max_retries:
                logger.error(f"Run {run_spec.run_id} ({regime or 'both'}, seed: {run_spec.seed}) failed after {max_retries} retries.")
                return {
                    "status": "failure",
                    "run_id": run_spec.run_id,
                    "regime": regime,
                    "seed": run_spec.seed,
                    "retries": attempt,
                    "error": str(e),
                }
    return {
        "status": "failure",
        "run_id": run_spec.run_id,
        "regime": regime,
        "seed": run_spec.seed,
        "retries": max_retries,
        "error": "Unknown error in _run_wrapper",
    }


def run_parallel(
    params_path: str,
    n_runs: int,
    max_retries: int,
    seed: Optional[int] = None,
    num_processes: Optional[int] = None,
    persistence_level: str = "full",
) -> Dict[str, Any]:
    """
    Orchestrates multiple independent simulation runs in parallel.
    Each run consists of one demographic generation and two textual replays.

    Args:
        params_path (str): Path to the YAML configuration file for the experiment.
        n_runs (int): The number of independent simulation runs to execute.
        max_retries (int): The maximum number of times to retry a failed run.
        seed (Optional[int]): An optional master seed for the random number generator.
                              If None, a default fixed seed is used.
        num_processes (Optional[int]): The number of parallel processes to use.
                                       If None, it defaults to the number of CPUs.
        persistence_level (str): The level of data persistence: 'minimal' or 'full'.

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

    # Generate the deterministic run plan
    run_plan = generate_run_plan(n_runs, seed=seed)

    successful_runs_count = 0
    failed_runs_count = 0
    failure_records: List[Dict[str, Any]] = []

    # Use default number of processes if not specified
    if num_processes is None:
        num_processes = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit each run to the executor. Each run will handle its own
        # demographic phase and both textual regimes, avoiding redundant work.
        futures = {}
        for run_spec in run_plan:
            future = executor.submit(
                _run_wrapper,
                params_path,
                run_spec,
                None,  # Run both regimes
                max_retries,
                persistence_level,
            )
            futures[future] = run_spec

        for future in as_completed(futures):
            run_spec = futures[future]
            try:
                result_data = future.result()
                if result_data["status"] == "success":
                    successful_runs_count += 1
                    logger.info(f"Run {run_spec.run_id} (seed: {run_spec.seed}) completed successfully.")
                else:
                    failed_runs_count += 1
                    failure_records.append({
                        "run_id": run_spec.run_id,
                        "seed": run_spec.seed,
                        "error": result_data["error"],
                        "attempt": result_data["retries"],
                    })
                    logger.error(f"Run {run_spec.run_id} (seed: {run_spec.seed}) failed: {result_data['error']}")
            except Exception as exc:
                failed_runs_count += 1
                failure_records.append({
                    "run_id": run_spec.run_id,
                    "seed": run_spec.seed,
                    "error": f"Unhandled exception: {str(exc)}",
                    "attempt": 0,
                })
                logger.critical(f"Run {run_spec.run_id} (seed: {run_spec.seed}) generated an unhandled exception: {exc}")

    return {
        "successful_runs": successful_runs_count,
        "failed_runs": failed_runs_count,
        "failure_records": failure_records,
        "total_runs_attempted": n_runs,
        "total_requested_runs": n_runs,
    }
