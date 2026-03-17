import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from pasim.execution.run_plan import RunSpec, generate_run_plan
from pasim.execution.runner import run_single
from pasim.io.results_aggregator import aggregate_results

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
                attempt=attempt,
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
    skip_run_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Orchestrates multiple independent simulation runs in parallel.
    Each run consists of one demographic generation and two textual replays.
    """
    if n_runs <= 0:
        return {"successful_runs": 0, "failed_runs": 0, "failure_records": [], "total_runs_attempted": 0}

    run_plan = generate_run_plan(n_runs, seed=seed, skip_run_ids=skip_run_ids)

    if not run_plan:
        return {
            "successful_runs": 0,
            "failed_runs": 0,
            "failure_records": [],
            "total_runs_attempted": 0,
            "total_requested_runs": n_runs,
        }

    successful_runs_count = 0
    failed_runs_count = 0
    failure_records: List[Dict[str, Any]] = []

    if num_processes is None:
        num_processes = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(_run_wrapper, params_path, spec, None, max_retries, persistence_level): spec for spec in run_plan}

        for future in as_completed(futures):
            run_spec = futures[future]
            try:
                result_data = future.result()
                success, record = _process_run_result(run_spec, result_data)
                if success:
                    successful_runs_count += 1
                else:
                    failed_runs_count += 1
                    failure_records.append(record)

                # Aggregation: Update results.csv after each run (if in minimal mode)
                aggregate_results(Path(params_path).parent)

            except Exception as exc:
                failed_runs_count += 1
                failure_records.append({"run_id": run_spec.run_id, "seed": run_spec.seed, "error": str(exc), "attempt": 0})
                logger.critical(f"Run {run_spec.run_id} (seed: {run_spec.seed}) unhandled exception: {exc}")

    return {
        "successful_runs": successful_runs_count,
        "failed_runs": failed_runs_count,
        "failure_records": failure_records,
        "total_runs_attempted": n_runs,
        "total_requested_runs": n_runs,
    }


def _process_run_result(run_spec: RunSpec, result_data: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    """Processes the result of a single run from the worker pool."""
    if result_data["status"] == "success":
        logger.info(f"Run {run_spec.run_id} (seed: {run_spec.seed}) completed successfully.")
        return True, {}

    record = {
        "run_id": run_spec.run_id,
        "seed": run_spec.seed,
        "error": result_data["error"],
        "attempt": result_data["retries"],
    }
    logger.error(f"Run {run_spec.run_id} (seed: {run_spec.seed}) failed: {result_data['error']}")
    return False, record
