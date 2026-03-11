import csv
import json
from pathlib import Path

import pytest

from pasim.execution.orchestrator import run_experiment

# Minimal configuration for testing
TEST_CONFIG = """
total_ticks: 10
text_length: 10
p_region_migration: 0.1
p_internal_relocation: 0.1
reputation_distribution:
  1: 0.2
  2: 0.2
  3: 0.2
  4: 0.2
  5: 0.2
pa_regime: insertion
pa_intervention_year: 5
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
  1: 20
  5: 40
n_runs: {n_runs}
seed: {seed}
"""


@pytest.fixture
def temp_experiment_dir(tmp_path: Path) -> Path:
    exp_dir = tmp_path / "experiments" / "deterministic_test"
    exp_dir.mkdir(parents=True)
    return exp_dir


def test_run_directory_alignment_full(temp_experiment_dir: Path):
    """Test 1: Run simulation with persistence=full and verify directories and metadata."""
    params_file = temp_experiment_dir / "params.yaml"
    n_runs = 4
    seed = 42
    params_file.write_text(TEST_CONFIG.format(n_runs=n_runs, seed=seed))

    run_experiment(str(params_file), persistence_level="full")

    runs_dir = temp_experiment_dir / "runs"
    assert runs_dir.exists()

    for i in range(n_runs):
        run_dir = runs_dir / f"run_{i}"
        assert run_dir.exists()
        assert (run_dir / "insertion").exists()
        assert (run_dir / "omission").exists()

        # Verify metadata
        for regime in ["insertion", "omission"]:
            metadata_path = run_dir / regime / "run_metadata.json"
            assert metadata_path.exists()
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                assert metadata["run_id"] == i
                assert metadata["pa_regime"] == regime


def test_minimal_mode_no_run_directories(temp_experiment_dir: Path):
    """Test 2: Minimal mode creates no run directories but creates results.csv."""
    params_file = temp_experiment_dir / "params.yaml"
    n_runs = 3
    seed = 123
    params_file.write_text(TEST_CONFIG.format(n_runs=n_runs, seed=seed))

    run_experiment(str(params_file), persistence_level="minimal")

    runs_dir = temp_experiment_dir / "runs"
    assert not runs_dir.exists()

    results_file = temp_experiment_dir / "results.csv"
    assert results_file.exists()

    # Results file check for 2 * runs rows
    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2 * n_runs


def test_deterministic_run_ids_parallel(temp_experiment_dir: Path):
    """Test 3: Deterministic Run IDs under parallelism."""
    params_file = temp_experiment_dir / "params.yaml"
    n_runs = 5
    seed = 999
    params_file.write_text(TEST_CONFIG.format(n_runs=n_runs, seed=seed))

    run_experiment(str(params_file), persistence_level="full")

    runs_dir = temp_experiment_dir / "runs"
    # Verify exactly run_0 to run_4 exist
    actual_runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])
    expected_runs = [f"run_{i}" for i in range(n_runs)]
    assert actual_runs == expected_runs


def test_results_file_consistency(temp_experiment_dir: Path):
    """Test 4: Results file consistency."""
    params_file = temp_experiment_dir / "params.yaml"
    n_runs = 3
    seed = 555
    params_file.write_text(TEST_CONFIG.format(n_runs=n_runs, seed=seed))

    run_experiment(str(params_file), persistence_level="minimal")

    results_file = temp_experiment_dir / "results.csv"
    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Verify run_ids 0..N-1
        run_ids = sorted(list(set(int(row["run_id"]) for row in rows)))
        assert run_ids == list(range(n_runs))

        # Each run appears twice (insertion and omission)
        from collections import Counter

        counts = Counter(int(row["run_id"]) for row in rows)
        for i in range(n_runs):
            assert counts[i] == 2

        # Total rows
        assert len(rows) == 2 * n_runs
