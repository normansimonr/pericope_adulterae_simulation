import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Union, cast

import yaml
from pydantic import ValidationError

from pasim.config.schema import SimulationConfig
from pasim.execution.parallel import run_parallel
from pasim.io.results_aggregator import aggregate_results


def _get_experiment_metadata_path(params_path_obj: Path) -> Path:
    """Determines the path to the experiment_metadata.json file."""
    experiment_dir = params_path_obj.parent
    return experiment_dir / "experiment_metadata.json"


def _init_experiment_metadata(params_file_path: Path, persistence_level: str) -> Dict[str, Any]:
    """Initializes and saves the initial experiment metadata."""
    metadata = {
        "experiment_id": params_file_path.parent.name,
        "params_path": str(params_file_path.name),
        "persistence_level": persistence_level,
        "execution_status": "started",
        "start_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "end_timestamp": None,
        "run_counts": {"successful": 0, "failed": 0, "retried": 0},
        "summary": None,
        "error_details": None,
        "simulation_config_snapshot": None,
        "parallelism_level_used": os.cpu_count() or 1,
    }
    experiment_metadata_path = _get_experiment_metadata_path(params_file_path)
    with open(experiment_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _load_and_validate_params(params_file_path: Path) -> Dict[str, Any]:
    """Loads and validates the experiment parameters from a YAML file."""
    with open(params_file_path, "r") as f:
        raw_params = yaml.safe_load(f)
    if not isinstance(raw_params, dict):
        raise ValueError("YAML file must contain a dictionary.")

    # Validate against SimulationConfig
    config = SimulationConfig(**raw_params)
    n_runs = raw_params.get("n_runs")
    if n_runs is None or not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError(f"Experiment params file '{params_file_path}' must contain a positive integer field 'n_runs'.")

    return {
        "config": config,
        "n_runs": n_runs,
        "max_retries": raw_params.get("max_retries", 1),
        "seed": raw_params.get("seed"),
    }


def _update_metadata_on_success(metadata: Dict[str, Any], experiment_metadata_path: Path, experiment_summary: Dict[str, Any]) -> None:
    """Updates and saves metadata after a successful experiment execution."""
    retried_runs_count = sum(record["attempt"] for record in experiment_summary.get("failure_records", []))
    final_status = "completed" if experiment_summary["failed_runs"] == 0 else "completed_with_failures"

    metadata.update({
        "execution_status": final_status,
        "end_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_counts": {
            "successful": int(experiment_summary["successful_runs"]),
            "failed": int(experiment_summary["failed_runs"]),
            "retried": int(retried_runs_count),
        },
        "summary": experiment_summary,
    })
    with open(experiment_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _update_metadata_on_error(metadata: Dict[str, Any], experiment_metadata_path: Path, error: Exception, status: str) -> None:
    """Updates and saves metadata after an experiment error."""
    try:
        with open(experiment_metadata_path, "r") as f:
            current_metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        current_metadata = metadata

    current_metadata.update({
        "execution_status": status,
        "end_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "error_details": repr(error),
    })
    with open(experiment_metadata_path, "w") as f:
        json.dump(current_metadata, f, indent=2)


def _get_completed_run_ids(experiment_root: Path) -> List[int]:
    """Reads results.csv to find IDs of already completed runs (must have both regimes)."""
    results_path = experiment_root / "results.csv"
    if not results_path.exists():
        return []

    import csv

    from pasim.io.results_aggregator import _coerce_row_types

    completed_regimes: Dict[int, Set[str]] = {}
    with open(results_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _coerce_row_types(row)
            run_id = int(row["run_id"])
            regime = row["regime"]
            completed_regimes.setdefault(run_id, set()).add(regime)

    # A run is only "done" if both regimes are present
    return [rid for rid, regimes in completed_regimes.items() if "insertion" in regimes and "omission" in regimes]


def run_experiment(params_path: Union[Path, str], persistence_level: str = "minimal") -> Dict[str, Any]:
    """
    Executes a complete experiment definition, orchestrating multiple parallel
    Monte Carlo runs based on the provided parameters file.
    Supports resuming from an existing results.csv file.
    """
    params_file_path = Path(params_path)
    if not params_file_path.is_file():
        raise FileNotFoundError(f"Experiment params file not found at: {params_file_path}")

    experiment_root = params_file_path.parent
    experiment_metadata_path = _get_experiment_metadata_path(params_file_path)
    metadata = _init_experiment_metadata(params_file_path, persistence_level)

    try:
        params = _load_and_validate_params(params_file_path)
        _update_initial_metadata(metadata, params, experiment_metadata_path)

        skip_run_ids = _handle_resumption_logic(metadata, experiment_root, persistence_level)

        experiment_summary = cast(
            Dict[str, Any],
            run_parallel(
                str(params_file_path),
                n_runs=params["n_runs"],
                max_retries=params["max_retries"],
                seed=params["seed"],
                persistence_level=persistence_level,
                skip_run_ids=skip_run_ids,
            ),
        )

        aggregate_results(experiment_root)
        _finalize_experiment(metadata, experiment_metadata_path, experiment_summary, experiment_root)

        return experiment_summary

    except yaml.YAMLError as e:
        _update_metadata_on_error(metadata, experiment_metadata_path, e, "errored_init")
        raise e
    except (ValueError, ValidationError) as e:
        _update_metadata_on_error(metadata, experiment_metadata_path, e, "errored_init")
        raise ValueError(f"Experiment initialization failed: {e}") from e
    except Exception as e:
        _update_metadata_on_error(metadata, experiment_metadata_path, e, "errored_runtime")
        raise


def _update_initial_metadata(metadata: Dict[str, Any], params: Dict[str, Any], metadata_path: Path) -> None:
    """Updates metadata with initial parameters and saves to disk."""
    metadata["simulation_config_snapshot"] = params["config"].model_dump(mode="json")
    metadata.update({
        "total_requested_runs": int(params["n_runs"]),
        "retry_policy": int(params["max_retries"]),
        "seed": int(params["seed"]) if params["seed"] is not None else None,
    })
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _handle_resumption_logic(metadata: Dict[str, Any], experiment_root: Path, persistence_level: str) -> List[int]:
    """Handles logic for resuming an experiment."""
    skip_run_ids = []
    if persistence_level == "minimal":
        skip_run_ids = _get_completed_run_ids(experiment_root)
        if skip_run_ids:
            print(f"Resuming experiment: found {len(skip_run_ids)} already completed runs in results.csv.")
            metadata["resumed_from_completed_runs"] = len(skip_run_ids)
    return skip_run_ids


def _finalize_experiment(metadata: Dict[str, Any], metadata_path: Path, summary: Dict[str, Any], experiment_root: Path) -> None:
    """Finalizes experiment summary, updates metadata, and prints output."""
    total_completed = len(_get_completed_run_ids(experiment_root))
    summary["total_successful_in_experiment"] = total_completed

    _update_metadata_on_success(metadata, metadata_path, summary)

    print("\nExperiment Summary:")
    print(f"Total Requested Runs: {metadata['total_requested_runs']}")
    print(f"Successfully Completed: {total_completed}")
    print(f"Runs completed in this session: {summary['successful_runs']}")
    print(f"Failed Runs: {summary['failed_runs']}")
    if summary["failed_runs"] > 0:
        print(f"Failure Details: {summary['failure_records']}")
