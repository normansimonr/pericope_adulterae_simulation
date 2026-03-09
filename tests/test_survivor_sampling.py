from pathlib import Path

import pytest
import yaml

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_generator import extract_genealogy_snapshot, run_genealogy_generator
from pasim.core.rng import RNGContext
from pasim.core.survivor_sampler import sample_survivors
from pasim.execution.runner import run_single

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
def sampling_config_path(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    p = exp_dir / "sampling_params.yaml"
    p.write_text(BASE_CONFIG)
    monkeypatch.chdir(tmp_path)
    return "experiments/sampling_params.yaml"


def test_sampling_eligibility():
    """Verify that sampled witnesses were born >= 300 and are alive at end."""
    params = yaml.safe_load(BASE_CONFIG)
    config = SimulationConfig(**params)
    seed = 42

    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(params, rng)
    snapshot = extract_genealogy_snapshot(state)

    result = sample_survivors(snapshot, seed, config.total_ticks)

    # Map for quick lookup
    node_map = {n.instance_id: n for n in snapshot.nodes}

    for sid in result.sampled_witness_ids:
        node = node_map[sid]
        assert node.birth_tick >= 300
        assert node.death_tick >= config.total_ticks


def test_sampling_determinism():
    """Run sampling twice with identical seed and genealogy. Assert identical IDs."""
    params = yaml.safe_load(BASE_CONFIG)
    config = SimulationConfig(**params)
    seed = 123

    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(params, rng)
    snapshot = extract_genealogy_snapshot(state)

    res1 = sample_survivors(snapshot, seed, config.total_ticks)
    res2 = sample_survivors(snapshot, seed, config.total_ticks)

    assert res1.sampled_witness_ids == res2.sampled_witness_ids
    assert res1.stratum_counts == res2.stratum_counts


def test_cross_regime_consistency(sampling_config_path):
    """Assert insertion survivors == omission survivors."""
    result = run_single(sampling_config_path, seed=456)
    assert len(result.survivor_sampling_result.sampled_witness_ids) > 0


def test_persistence_filtering(sampling_config_path, tmp_path):
    """Verify that output files contain ONLY sampled witnesses."""
    res = run_single(sampling_config_path, seed=789)

    # Based on resolve_run_directory implementation:
    # experiment_dir = params_path.parent -> "experiments"
    # runs_dir = experiment_dir / "runs" -> "experiments/runs"
    runs_dir = Path("experiments/runs")
    run_nums = [int(d.name[4:]) for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    latest_run = max(run_nums)
    run_dir = runs_dir / f"run_{latest_run}"

    survivor_ids = set(res.survivor_sampling_result.sampled_witness_ids)

    for regime in ["insertion", "omission"]:
        tsv_path = run_dir / regime / "instance_texts.tsv"
        assert tsv_path.exists()
        assert (run_dir / regime / "witnesses.parquet").exists()

        with open(tsv_path, "r") as f:
            lines = f.readlines()
            if len(survivor_ids) > 0:
                assert len(lines) == len(survivor_ids) + 1
                for line in lines[1:]:
                    sid = line.split("\t")[0]
                    assert sid in survivor_ids


def test_warning_behavior():
    """Construct a small artificial genealogy that cannot satisfy quotas. Verify warnings."""
    from pasim.core.genealogy_snapshot import GenealogyNode, GenealogySnapshot
    from pasim.core.state import Material, Region, Script

    nodes = []
    for i in range(10):
        nodes.append(
            GenealogyNode(
                instance_id=f"node_{i}",
                witness_id=f"w_{i}",
                manuscript_id=f"m_{i}",
                birth_tick=700,
                death_tick=1000,
                parent_ids=[],
                region=Region.ASIA_MINOR,
                material=Material.PAPYRUS,
                script=Script.UNCIAL,
                reputation=3,
                location=(0, 0),
            )
        )

    snapshot = GenealogySnapshot(nodes=nodes)
    res = sample_survivors(snapshot, base_seed=1, total_ticks=1000)

    assert res.actual_sample_size == 10
    assert len(res.warnings) > 0
    assert "AsiaMinor_650+" in res.deficit_per_stratum
