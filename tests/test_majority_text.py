import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from pasim.analysis.majority_text import compute_majority_text
from pasim.execution.runner import run_single


def test_mode_calculation():
    """Verify correct majority selection."""
    genomes = [np.array([0, 1, 1, 2], dtype=np.int16), np.array([0, 1, 2, 2], dtype=np.int16), np.array([0, 1, 1, 3], dtype=np.int16)]
    # Segment 0: [0, 0, 0] -> mode 0
    # Segment 1: [1, 1, 1] -> mode 1
    # Segment 2: [1, 2, 1] -> mode 1
    # Segment 3: [2, 2, 3] -> mode 2
    majority = compute_majority_text(genomes)
    assert majority == [0, 1, 1, 2]


def test_tie_break_determinism():
    """Verify deterministic tie-breaking: choose the smallest value."""
    genomes = [np.array([1, 1], dtype=np.int16), np.array([2, 2], dtype=np.int16)]
    # Segment 0: [1, 2] -> tie, choose 1
    # Segment 1: [1, 2] -> tie, choose 1
    majority = compute_majority_text(genomes)
    assert majority == [1, 1]

    genomes2 = [np.array([2, 2], dtype=np.int16), np.array([1, 1], dtype=np.int16)]
    majority2 = compute_majority_text(genomes2)
    assert majority2 == [1, 1]


def test_genome_length_integrity():
    """Verify that output length equals genome segment count."""
    text_length = 50
    genomes = [np.random.randint(0, 6, size=text_length, dtype=np.int16) for _ in range(10)]
    majority = compute_majority_text(genomes)
    assert len(majority) == text_length


def test_empty_survivor_case():
    """Verify that empty survivor list returns empty majority text."""
    majority = compute_majority_text([])
    assert majority == []


BASE_CONFIG = """
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
def majority_config_path(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    p = exp_dir / "majority_params.yaml"
    p.write_text(BASE_CONFIG)
    monkeypatch.chdir(tmp_path)
    return "experiments/majority_params.yaml"


def test_cross_regime_independence(majority_config_path):
    """Confirm both produce majority files and that values may differ (or at least exist)."""
    result = run_single(majority_config_path, seed=42)
    run_dir = Path("experiments/runs/1")

    for regime in ["insertion", "omission"]:
        majority_path = run_dir / regime / "majority_text.json"
        assert majority_path.exists()
        with open(majority_path, "r") as f:
            data = json.load(f)
            assert data["segment_count"] == result.config.text_length
            assert len(data["majority_segments"]) == result.config.text_length

    # Compare them - they should differ because insertion/omission initial texts differ
    with open(run_dir / "insertion" / "majority_text.json", "r") as f:
        ins_data = json.load(f)
    with open(run_dir / "omission" / "majority_text.json", "r") as f:
        om_data = json.load(f)

    assert ins_data["majority_segments"] != om_data["majority_segments"]


def test_majority_text_determinism(majority_config_path):
    """Run same experiment twice with identical seed. Assert identical majority_text.json."""
    run_single(majority_config_path, seed=123)
    run_dir1 = Path("experiments/runs/1")

    # Clear runs or expect run 2
    run_single(majority_config_path, seed=123)
    run_dir2 = Path("experiments/runs/2")

    for regime in ["insertion", "omission"]:
        with open(run_dir1 / regime / "majority_text.json", "r") as f:
            data1 = json.load(f)
        with open(run_dir2 / regime / "majority_text.json", "r") as f:
            data2 = json.load(f)
        assert data1 == data2


def test_empty_survivor_persistence(tmp_path, monkeypatch):
    """Verify persistence behavior when zero witnesses survive."""
    # Create a config with total_ticks < 300 so no one is eligible
    config_data = yaml.safe_load(BASE_CONFIG)
    config_data["total_ticks"] = 100
    config_data["pa_intervention_year"] = 1  # must be <= total_ticks

    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    p = exp_dir / "empty_params.yaml"
    p.write_text(yaml.dump(config_data))
    monkeypatch.chdir(tmp_path)

    run_single("experiments/empty_params.yaml", seed=999)
    run_dir = Path("experiments/runs/1")

    for regime in ["insertion", "omission"]:
        majority_path = run_dir / regime / "majority_text.json"
        assert majority_path.exists()
        with open(majority_path, "r") as f:
            data = json.load(f)
            assert data["segment_count"] == 0
            assert data["majority_segments"] == []
