import csv

import pytest
import yaml

from pasim.execution.orchestrator import run_experiment

BASE_CONFIG = """
n_runs: 2
total_ticks: 1000
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
pa_intervention_year: 300
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
  1: 120
  300: 600
  700: 1200
"""


@pytest.fixture
def experiment_dir(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments" / "test_persistence"
    exp_dir.mkdir(parents=True, exist_ok=True)
    p = exp_dir / "params.yaml"
    p.write_text(BASE_CONFIG)
    monkeypatch.chdir(tmp_path)
    return exp_dir


def test_minimal_persistence(experiment_dir):
    """Verify results.csv exists and no per-run directories created in minimal mode."""
    params_path = experiment_dir / "params.yaml"
    run_experiment(params_path, persistence_level="minimal")

    assert (experiment_dir / "results.csv").exists()
    assert not (experiment_dir / "runs").exists()
    assert not (experiment_dir / "temp_results").exists()

    # Check results.csv content
    with open(experiment_dir / "results.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # 2 runs * 2 regimes = 4 rows
        assert len(rows) == 4
        # Verify columns
        for row in rows:
            assert "run_id" in row
            assert "run_seed" in row
            assert "regime" in row
            assert "total_manuscripts_spawned" in row
            assert "majority_text" in row


def test_full_persistence(experiment_dir):
    """Verify run artefacts exist and results.csv also exists in full mode."""
    # Reset dir for fresh test
    if (experiment_dir / "results.csv").exists():
        (experiment_dir / "results.csv").unlink()

    params_path = experiment_dir / "params.yaml"
    run_experiment(params_path, persistence_level="full")

    assert (experiment_dir / "results.csv").exists()
    assert (experiment_dir / "runs").exists()
    assert (experiment_dir / "runs" / "run_0").exists()
    assert (experiment_dir / "runs" / "run_1").exists()

    # Verify some artefact in run 0
    assert (experiment_dir / "runs" / "run_0" / "genealogy.json").exists()


def test_majority_text_string_serialization():
    """Verify majority_text stored as string and leading zeros preserved."""
    from pasim.io.results_writer import serialize_majority_text

    # Case with leading zeros
    segments = [0, 0, 1, 2, 3, 0]
    serialized = serialize_majority_text(segments)
    assert serialized == "001230"
    assert isinstance(serialized, str)

    # Case with only zeros
    segments = [0, 0, 0]
    serialized = serialize_majority_text(segments)
    assert serialized == "000"


def test_empty_survivor_aggregation(tmp_path, monkeypatch):
    """Verify row contains majority_text = "" when zero witnesses survive."""
    exp_dir = tmp_path / "experiments" / "test_empty"
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_data = yaml.safe_load(BASE_CONFIG)
    config_data["total_ticks"] = 100
    config_data["pa_intervention_year"] = 1
    config_data["n_runs"] = 1

    p = exp_dir / "params.yaml"
    p.write_text(yaml.dump(config_data))
    monkeypatch.chdir(tmp_path)

    run_experiment(p, persistence_level="minimal")

    results_path = exp_dir / "results.csv"
    assert results_path.exists()

    with open(results_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2  # 1 run * 2 regimes
        for row in rows:
            assert row["majority_text"] == ""


def test_parallel_aggregation_determinism(experiment_dir):
    """Verify aggregation produces sorted output."""
    params_path = experiment_dir / "params.yaml"
    run_experiment(params_path, persistence_level="minimal")

    results_path = experiment_dir / "results.csv"
    with open(results_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Sorted by run_id, then regime
    assert rows[0]["run_id"] == "0"
    assert rows[0]["regime"] == "insertion"
    assert rows[1]["run_id"] == "0"
    assert rows[1]["regime"] == "omission"
    assert rows[2]["run_id"] == "1"
    assert rows[2]["regime"] == "insertion"
    assert rows[3]["run_id"] == "1"
    assert rows[3]["regime"] == "omission"
