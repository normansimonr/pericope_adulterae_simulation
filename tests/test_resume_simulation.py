import csv

import pytest
import yaml

from pasim.execution.orchestrator import run_experiment

BASE_CONFIG = """
n_runs: 5
total_ticks: 50
text_length: 10
reputation_distribution:
  1: 1.0
  2: 0.0
  3: 0.0
  4: 0.0
  5: 0.0
pa_regime: insertion
pa_intervention_year: 25
pa_intervention_region: "Asia Minor"
pa_innovator_reputation: 5.0
material_transitions:
  - start_tick: 0
    distribution:
      parchment: 1.0
script_transitions:
  - start_tick: 0
    distribution:
      uncial: 1.0
demand_schedule:
  1: 5
"""


@pytest.fixture
def resume_experiment_dir(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments" / "test_resume"
    exp_dir.mkdir(parents=True, exist_ok=True)
    p = exp_dir / "params.yaml"
    p.write_text(BASE_CONFIG)
    monkeypatch.chdir(tmp_path)
    return exp_dir


def test_resume_simulation_incremental(resume_experiment_dir):
    """Verify that increasing n_runs resumes and only executes new runs."""
    params_path = resume_experiment_dir / "params.yaml"

    # First run: 5 runs
    run_experiment(params_path, persistence_level="minimal")

    results_path = resume_experiment_dir / "results.csv"
    with open(results_path, "r") as f:
        rows_initial = list(csv.DictReader(f))
    assert len(rows_initial) == 10  # 5 runs * 2 regimes

    # Increase n_runs to 8
    config = yaml.safe_load(BASE_CONFIG)
    config["n_runs"] = 8
    params_path.write_text(yaml.dump(config))

    # Second run: should only do 3 more runs
    summary = run_experiment(params_path, persistence_level="minimal")
    assert summary["successful_runs"] == 3
    assert summary["total_successful_in_experiment"] == 8

    with open(results_path, "r") as f:
        rows_final = list(csv.DictReader(f))
    assert len(rows_final) == 16  # 8 runs * 2 regimes

    # Verify run_ids are 0 to 7
    run_ids = sorted(list(set(int(r["run_id"]) for r in rows_final)))
    assert run_ids == list(range(8))


def test_resume_simulation_no_work(resume_experiment_dir):
    """Verify that running again with same n_runs does nothing."""
    params_path = resume_experiment_dir / "params.yaml"

    run_experiment(params_path, persistence_level="minimal")

    # Run again with same config
    summary = run_experiment(params_path, persistence_level="minimal")
    assert summary["successful_runs"] == 0
    assert summary["total_successful_in_experiment"] == 5


def test_resume_seed_consistency(resume_experiment_dir):
    """Verify that seeds remain consistent when resuming."""
    params_path = resume_experiment_dir / "params.yaml"

    # Run 5 runs with a fixed seed
    config = yaml.safe_load(BASE_CONFIG)
    config["seed"] = 12345
    params_path.write_text(yaml.dump(config))

    run_experiment(params_path, persistence_level="minimal")

    results_path = resume_experiment_dir / "results.csv"
    with open(results_path, "r") as f:
        rows_first = {(int(r["run_id"]), r["regime"]): r["run_seed"] for r in csv.DictReader(f)}

    # Delete results.csv and run 3 then 2
    results_path.unlink()

    config["n_runs"] = 3
    params_path.write_text(yaml.dump(config))
    run_experiment(params_path, persistence_level="minimal")

    config["n_runs"] = 5
    params_path.write_text(yaml.dump(config))
    run_experiment(params_path, persistence_level="minimal")

    with open(results_path, "r") as f:
        rows_second = {(int(r["run_id"]), r["regime"]): r["run_seed"] for r in csv.DictReader(f)}

    assert rows_first == rows_second
