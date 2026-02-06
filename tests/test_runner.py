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
persecution_events: []
material_transitions:
  - start_tick: 0
    distribution:
      papyrus: 1.0
script_transitions:
  - start_tick: 0
    distribution:
      uncial: 1.0
demand_schedule:
  0:
    Asia Minor: 1
death_ticks: [1, 2]
n_runs: 3
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

    summary = run_parallel(str(params_path))

    # Assert summary is correct for a successful run
    assert summary["total_runs"] == n_runs
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
    with patch("pasim.execution.parallel.run_single", side_effect=ValueError("Test failure")):
        summary = run_parallel(str(params_path))

    # Assert summary reflects the failed runs and retries
    assert summary["total_runs"] == n_runs
    assert summary["successful_runs"] == 0
    assert summary["failed_runs"] == n_runs

    # Each of the n_runs should have (max_retries + 1) attempts
    total_attempts = n_runs * (max_retries + 1)
    assert len(summary["failure_records"]) == total_attempts

    # Check the details of one of the failure records
    first_failure = summary["failure_records"][0]
    assert first_failure["run_index"] == 0
    assert first_failure["attempt"] == 1
    assert "ValueError('Test failure')" in first_failure["exception"]

    last_failure = summary["failure_records"][-1]
    assert last_failure["run_index"] == n_runs - 1
    assert last_failure["attempt"] == max_retries + 1


# Test for the new run_experiment entrypoint
def test_run_experiment_successful_execution(temp_parallel_experiment_folder: Path):
    """
    Tests that run_experiment orchestrates parallel runs successfully and returns a correct summary.
    """
    params_path = temp_parallel_experiment_folder / "params.yaml"

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    n_runs = params["n_runs"]

    # Call run_experiment, which internally uses run_parallel
    summary = run_experiment(str(params_path))

    # Assert summary is correct for a successful experiment
    assert summary["total_runs"] == n_runs
    assert summary["successful_runs"] == n_runs
    assert summary["failed_runs"] == 0
    assert len(summary["failure_records"]) == 0

    # Assert that n_runs directories were created by the underlying parallel execution
    runs_dir = temp_parallel_experiment_folder / "runs"
    assert runs_dir.is_dir()
    created_run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(created_run_dirs) == n_runs
