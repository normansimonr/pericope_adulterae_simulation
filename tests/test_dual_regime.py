from pathlib import Path

import numpy as np
import pytest

from pasim.execution.runner import run_single

# Minimal configuration for testing dual-regime
DUAL_REGIME_CONFIG = """
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
"""


@pytest.fixture
def temp_experiment_dir(tmp_path: Path) -> Path:
    exp_dir = tmp_path / "experiments" / "dual_regime_test"
    exp_dir.mkdir(parents=True)
    params_file = exp_dir / "params.yaml"
    params_file.write_text(DUAL_REGIME_CONFIG)
    return exp_dir


def test_dual_regime_structure(temp_experiment_dir: Path):
    """Verifies that a single run produces the expected dual-regime directory structure."""
    params_path = temp_experiment_dir / "params.yaml"
    seed = 42

    run_single(str(params_path), seed=seed)

    # Check that the run directory was created (should be '1')
    run_dir = temp_experiment_dir / "runs" / "1"
    assert run_dir.exists()

    # Check for shared files in the run root
    assert (run_dir / "genealogy.json").exists()
    assert (run_dir / "genealogy_snapshot.json").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "demographic_metadata.json").exists()

    # Verify that only ONE genealogy file and ONE snapshot file were created
    # by checking the count of files matching those names in the run directory.
    # (Since we just checked their existence, we just need to ensure no duplicates in subdirs)
    all_files = list(run_dir.rglob("*"))
    genealogy_files = [f for f in all_files if f.name == "genealogy.json"]
    snapshot_files = [f for f in all_files if f.name == "genealogy_snapshot.json"]

    assert len(genealogy_files) == 1
    assert len(snapshot_files) == 1

    # Check for regime subdirectories
    assert (run_dir / "insertion").exists()
    assert (run_dir / "omission").exists()

    # Check for regime-specific files
    for regime in ["insertion", "omission"]:
        regime_dir = run_dir / regime
        assert (regime_dir / "run_metadata.json").exists()
        assert (regime_dir / "instance_texts.tsv").exists()
        # Verify no genealogy file in the regime subdir
        assert not (regime_dir / "genealogy.json").exists()
        assert not (regime_dir / "genealogy_snapshot.json").exists()


def test_dual_regime_identical_genealogy(temp_experiment_dir: Path):
    """Verifies that both regimes share the exact same genealogy (graph structure)."""
    params_path = temp_experiment_dir / "params.yaml"
    seed = 123

    result = run_single(str(params_path), seed=seed)

    # The genealogy in the result object should be the same for both
    # (it's stored once in SimulationResult.graph)
    # But we also verify that the replays were performed on the same snapshot nodes
    insertion_texts = result.replays["insertion"].instance_texts
    omission_texts = result.replays["omission"].instance_texts

    assert set(insertion_texts.keys()) == set(omission_texts.keys())
    assert len(insertion_texts) == result.graph.number_of_nodes()


def test_dual_regime_different_results(temp_experiment_dir: Path):
    """Verifies that textual results differ between regimes."""
    params_path = temp_experiment_dir / "params.yaml"
    seed = 999

    result = run_single(str(params_path), seed=seed)

    insertion_texts = result.replays["insertion"].instance_texts
    omission_texts = result.replays["omission"].instance_texts

    # They should differ significantly because of the different starting point
    # and the PA intervention (insertion: 0->1, omission: 1->0)

    # Find the autograph (root node)
    roots = [n for n, d in result.graph.in_degree() if d == 0]
    assert len(roots) == 1

    # Autograph for insertion should be all 0s (default in make_initial_text for insertion)
    # Wait, make_initial_text uses default legal segment value.
    # Actually TextReplayEngine._process_node for root node:
    # if not node.parent_ids: text = make_initial_text(self.config)

    # Let's check what make_initial_text does.
    # In my experience, if insertion is active, it might be 0s, and omission 1s.

    # Regardless, the texts at the leaves should be different.
    leaves = [n for n, d in result.graph.out_degree() if d == 0]
    differences = 0
    for leaf_id in leaves:
        if not np.array_equal(insertion_texts[leaf_id], omission_texts[leaf_id]):
            differences += 1

    assert differences > 0, "Texts should differ between regimes"


def test_dual_regime_reproducibility(temp_experiment_dir: Path):
    """Verifies that the same seed produces identical dual-regime results."""
    params_path = temp_experiment_dir / "params.yaml"
    seed = 777

    # Run 1
    result1 = run_single(str(params_path), seed=seed)

    # Run 2 (will be in directory '2')
    result2 = run_single(str(params_path), seed=seed)

    # Compare graphs
    assert list(result1.graph.nodes()) == list(result2.graph.nodes())
    assert list(result1.graph.edges()) == list(result2.graph.edges())

    # Compare replay results
    for regime in ["insertion", "omission"]:
        texts1 = result1.replays[regime].instance_texts
        texts2 = result2.replays[regime].instance_texts
        for node_id in texts1:
            assert np.array_equal(texts1[node_id], texts2[node_id])


def test_replay_seed_derivation():
    """Verifies that replay seeds are deterministic and differ by regime."""
    from pasim.execution.runner import derive_replay_seed

    seed = 100
    seed_ins = derive_replay_seed(seed, "insertion")
    seed_om = derive_replay_seed(seed, "omission")

    assert seed_ins != seed_om
    assert seed_ins == derive_replay_seed(seed, "insertion")
    assert seed_om == derive_replay_seed(seed, "omission")
