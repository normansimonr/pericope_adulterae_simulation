from pathlib import Path

import pytest
import yaml
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
n_runs: 3 # Number of runs for parallel execution
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


def test_run_parallel_creates_multiple_run_directories(temp_parallel_experiment_folder: Path):
    params_path = temp_parallel_experiment_folder / "params.yaml"

    # Load n_runs from the YAML to assert correctly
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    n_runs = params["n_runs"]

    run_parallel(str(params_path))

    # Assert that n_runs directories were created
    runs_dir = temp_parallel_experiment_folder / "runs"
    assert runs_dir.is_dir()

    created_run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(created_run_dirs) == n_runs

    # Verify naming convention (1, 2, ..., n_runs)
    expected_run_names = {str(i) for i in range(1, n_runs + 1)}
    actual_run_names = {d.name for d in created_run_dirs}
    assert actual_run_names == expected_run_names

    # Optional: basic check for files existence in one of the run directories
    # Pick the first one as an example
    if n_runs > 0:
        first_run_dir = runs_dir / "1"
        expected_files = [
            "config.yaml",
            "run_metadata.json",
            "genealogy.json",
            "instances.json",
            "manuscripts.json",
            "instance_texts.tsv",
            "telemetry.json",
            "events.log",
        ]
        for file_name in expected_files:
            assert (first_run_dir / file_name).is_file(), f"Missing file: {file_name}"
