import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, Union, cast

import yaml

from pasim.execution.parallel import run_parallel


def _get_experiment_metadata_path(params_path_obj: Path) -> Path:
    """Determines the path to the experiment_metadata.json file."""
    experiment_dir = params_path_obj.parent
    return experiment_dir / "experiment_metadata.json"


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

    experiment_metadata_path = _get_experiment_metadata_path(params_file_path)

    # Initial metadata structure (will be updated after params are loaded)
    metadata: Dict[str, Any] = {
        "experiment_id": params_file_path.parent.name,
        "params_path": str(params_file_path.name),  # Store relative path (just the filename)
        "total_requested_runs": None,  # Placeholder
        "parallelism_level_used": os.cpu_count() or 1,
        "retry_policy": None,  # Placeholder
        "seed": None,  # Placeholder
        "execution_status": "started",
        "start_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_counts": {"successful": 0, "failed": 0, "retried": 0},
        "end_timestamp": None,
        "summary": None,
        "error_details": None,
    }
    # Write initial metadata (started state) - always exists as soon as possible
    with open(experiment_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        # Load params and update metadata with config-dependent values
        with open(params_file_path, "r") as f:
            params = yaml.safe_load(f)

        n_runs = params.get("n_runs")
        if n_runs is None or not isinstance(n_runs, int) or n_runs <= 0:
            raise ValueError(f"Experiment params file '{params_file_path}' must contain a positive integer field 'n_runs'.")

        max_retries = params.get("max_retries", 1)
        base_seed = params.get("seed")

        metadata.update({
            "total_requested_runs": n_runs,
            "retry_policy": max_retries,
            "seed": base_seed,
        })
        # Overwrite metadata with updated config info
        with open(experiment_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Invoke the parallel orchestrator
        experiment_summary = cast(Dict[str, Any], run_parallel(str(params_file_path), base_seed=base_seed))

        # Calculate retried_runs
        retried_runs_count = sum(1 for record in experiment_summary["failure_records"] if record["attempt"] > 1)

        # Final metadata update (completed state)
        final_status = "completed"
        if experiment_summary["failed_runs"] > 0:
            final_status = "completed_with_failures"

        metadata.update({
            "execution_status": final_status,
            "end_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_counts": {
                "successful": experiment_summary["successful_runs"],
                "failed": experiment_summary["failed_runs"],
                "retried": retried_runs_count,
            },
            "summary": experiment_summary,
        })
        with open(experiment_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("\nExperiment Summary:")
        print(f"Total Runs: {metadata['total_requested_runs']}")
        print(f"Successful Runs: {metadata['run_counts']['successful']}")
        print(f"Failed Runs: {metadata['run_counts']['failed']}")
        if metadata["run_counts"]["failed"] > 0:
            print(f"Failure Details: {metadata['summary']['failure_records']}")

        return experiment_summary

    except Exception as e:
        # If any error occurs (including params loading failure), update metadata
        # Load the last written metadata to ensure we don't overwrite with old placeholders
        try:
            with open(experiment_metadata_path, "r") as f:
                current_metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_metadata = metadata  # Fallback to our initial in-memory dict if file was never written or corrupted

        current_metadata.update({
            "execution_status": "errored",
            "end_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "error_details": repr(e),
        })
        with open(experiment_metadata_path, "w") as f:
            json.dump(current_metadata, f, indent=2)
        raise  # Re-raise the exception after logging it
