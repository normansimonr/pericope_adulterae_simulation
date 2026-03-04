import json  # New import for json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pasim.execution.orchestrator import run_experiment
from pasim.execution.parallel import run_parallel

# Minimal configuration for fast testing of parallel runs
MINIMAL_PARALLEL_PARAMS_YAML = """
total_ticks: 2
text_length: 5
p_region_migration: 0.0
p_internal_relocation: 0.0
reputation_distribution:
  1: 0.2
  2: 0.2
  3: 0.2
  4: 0.2
  5: 0.2
pa_regime: insertion
pa_intervention_year: 1
pa_intervention_region: "Asia Minor"
pa_innovator_reputation: 5.0
persecutions: []
material_transitions:
  - start_tick: 0
    distribution:
      papyrus: 1.0
script_transitions:
  - start_tick: 0
    distribution:
      uncial: 1.0
demand_schedule:
  0: 1
n_runs: 3
max_retries: 1 # Add max_retries for testing
"""


@pytest.fixture
def temp_parallel_experiment_folder(tmp_path: Path) -> Path:
    """
    Creates a temporary experiment folder structure with a minimal params.yaml for parallel runs.
    """
    exp_name = "test_parallel_experiment"
    exp_dir = tmp_path / "experiments" / exp_name
    exp_dir.mkdir(parents=True)
    params_file = exp_dir / "params.yaml"
    params_file.write_text(MINIMAL_PARALLEL_PARAMS_YAML)
    return exp_dir


def test_run_parallel_successful_execution(temp_parallel_experiment_folder: Path):
    """
    Tests that run_parallel completes successfully and returns a correct summary.
    """
    params_path = temp_parallel_experiment_folder / "params.yaml"

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    n_runs = params["n_runs"]

    summary = run_parallel(str(params_path), n_runs=n_runs, max_retries=params["max_retries"])

    # Assert summary is correct for a successful run
    assert summary["total_runs_attempted"] == n_runs
    assert summary["successful_runs"] == n_runs
    assert summary["failed_runs"] == 0
    assert len(summary["failure_records"]) == 0

    # Assert that n_runs directories were created
    runs_dir = temp_parallel_experiment_folder / "runs"
    assert runs_dir.is_dir()
    created_run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(created_run_dirs) == n_runs


def test_run_parallel_failure_handling(temp_parallel_experiment_folder: Path):
    """
    Tests that run_parallel correctly handles failures and retries.
    """
    params_path = temp_parallel_experiment_folder / "params.yaml"

    # Modify params to include retries for this specific test
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    params["n_runs"] = 2
    params["max_retries"] = 2
    n_runs = params["n_runs"]
    max_retries = params["max_retries"]

    with open(params_path, "w") as f:
        yaml.dump(params, f)

    # Mock run_single to always fail
    with patch("pasim.execution.parallel.run_single", side_effect=ValueError("Run failure")):
        summary = run_parallel(str(params_path), n_runs=n_runs, max_retries=max_retries)

    # Assert summary reflects the failed runs and retries
    assert summary["total_runs_attempted"] == n_runs
    assert summary["successful_runs"] == 0
    assert summary["failed_runs"] == n_runs

    assert len(summary["failure_records"]) == n_runs

    # Check the details of one of the failure records
    first_failure = summary["failure_records"][0]
    assert "seed" in first_failure
    assert first_failure["error"] == "Run failure"
    assert first_failure["attempt"] == max_retries

    last_failure = summary["failure_records"][-1]
    assert "seed" in last_failure
    assert last_failure["error"] == "Run failure"
    assert last_failure["attempt"] == max_retries


# Test for the new run_experiment entrypoint
def test_run_experiment_successful_execution(temp_parallel_experiment_folder: Path):
    """
    Tests that run_experiment orchestrates parallel runs successfully and returns a correct summary.
    Also verifies experiment metadata.
    """
    params_path = temp_parallel_experiment_folder / "params.yaml"

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    n_runs = params["n_runs"]
    max_retries = params["max_retries"]  # Should be 1 from MINIMAL_PARALLEL_PARAMS_YAML
    base_seed = params.get("seed")  # Should be None

    summary = run_experiment(str(params_path))

    # Assert summary is correct for a successful experiment
    assert summary["total_runs_attempted"] == n_runs
    assert summary["successful_runs"] == n_runs
    assert summary["failed_runs"] == 0
    assert len(summary["failure_records"]) == 0

    # Assert experiment_metadata.json exists and is correct
    metadata_path = temp_parallel_experiment_folder / "experiment_metadata.json"
    assert metadata_path.is_file()
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert metadata["experiment_id"] == temp_parallel_experiment_folder.name
    assert metadata["params_path"] == params_path.name
    assert metadata["total_requested_runs"] == n_runs
    assert metadata["parallelism_level_used"] == (os.cpu_count() or 1)
    assert metadata["retry_policy"] == max_retries
    assert metadata["seed"] == base_seed
    assert metadata["execution_status"] == "completed"
    assert metadata["start_timestamp"] is not None
    assert metadata["end_timestamp"] is not None
    assert metadata["run_counts"]["successful"] == n_runs
    assert metadata["run_counts"]["failed"] == 0
    assert metadata["run_counts"]["retried"] == 0
    assert metadata["summary"] == summary


def test_run_experiment_failure_metadata(temp_parallel_experiment_folder: Path):
    """
    Tests that run_experiment correctly records metadata for an experiment with run-level failures.
    """
    params_path = temp_parallel_experiment_folder / "params.yaml"

    # Modify params to include retries for this specific test
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    params["n_runs"] = 2
    params["max_retries"] = 1  # 1 retry, so 2 attempts per run
    n_runs = params["n_runs"]
    max_retries = params["max_retries"]

    with open(params_path, "w") as f:
        yaml.dump(params, f)

    # Mock _run_single_with_retry to always fail
    with patch("pasim.execution.parallel.run_single", side_effect=ValueError("Run failure")):
        summary = run_experiment(str(params_path))

    # Assert summary reflects the failed runs
    assert summary["total_runs_attempted"] == n_runs
    assert summary["successful_runs"] == 0
    assert summary["failed_runs"] == n_runs
    assert len(summary["failure_records"]) == n_runs

    # Assert experiment_metadata.json exists and is correct for failures
    metadata_path = temp_parallel_experiment_folder / "experiment_metadata.json"
    assert metadata_path.is_file()
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert metadata["experiment_id"] == temp_parallel_experiment_folder.name
    assert metadata["execution_status"] == "completed_with_failures"
    assert metadata["start_timestamp"] is not None
    assert metadata["end_timestamp"] is not None
    assert metadata["run_counts"]["successful"] == 0
    assert metadata["run_counts"]["failed"] == n_runs
    assert metadata["run_counts"]["retried"] == n_runs * max_retries  # 2 runs * 1 retry


def test_run_experiment_catastrophic_failure_metadata(temp_parallel_experiment_folder: Path):
    """
    Tests that run_experiment correctly records metadata for catastrophic errors
    (e.g., during parameter parsing or before parallel execution starts).
    """
    params_path = temp_parallel_experiment_folder / "params.yaml"

    # Corrupt the params file to cause a catastrophic failure during parsing
    with open(params_path, "w") as f:
        f.write("invalid yaml: -")

    # Expect a ValueError to be raised by run_experiment due to bad YAML
    with pytest.raises(yaml.YAMLError):
        run_experiment(str(params_path))

    # Assert experiment_metadata.json exists and shows 'errored' status
    metadata_path = temp_parallel_experiment_folder / "experiment_metadata.json"
    assert metadata_path.is_file()
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert metadata["experiment_id"] == temp_parallel_experiment_folder.name
    assert metadata["execution_status"] == "errored_init"  # Changed from "errored"
    assert metadata["start_timestamp"] is not None
    assert metadata["end_timestamp"] is not None
    assert "error_details" in metadata
    assert "ScannerError" in metadata["error_details"]
    assert metadata["run_counts"]["successful"] == 0
    assert metadata["run_counts"]["failed"] == 0
    assert metadata["run_counts"]["retried"] == 0
    assert metadata["summary"] is None  # Summary should not be available for catastrophic failure
