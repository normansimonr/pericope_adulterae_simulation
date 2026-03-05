import numpy as np
import yaml

from pasim.execution.runner import run_single


def test_full_simulation_reproducibility(tmp_path):
    """
    Verify that two full simulation runs with the same seed produce identical results.
    """
    # Ensure experiments directory exists for run_single validation
    exp_dir = tmp_path / "experiments" / "reproducibility_test"
    exp_dir.mkdir(parents=True)
    params_path = exp_dir / "params.yaml"

    config_data = {
        "total_ticks": 10,
        "text_length": 20,
        "p_region_migration": 0.1,
        "p_internal_relocation": 0.1,
        "reputation_distribution": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "pa_regime": "insertion",
        "pa_intervention_year": 1,
        "pa_intervention_region": "Asia Minor",
        "pa_innovator_reputation": 5.0,
        "persecutions": [{"start_tick": 5, "end_tick": 5, "regions": ["Egypt"], "kill_proportion": 0.5}],
        "material_transitions": [
            {"start_tick": 0, "distribution": {"papyrus": 0.5, "parchment": 0.5}},
            {"start_tick": 7, "distribution": {"papyrus": 0.1, "parchment": 0.9}},
        ],
        "script_transitions": [
            {"start_tick": 0, "distribution": {"uncial": 1.0}},
            {"start_tick": 8, "distribution": {"uncial": 0.5, "minuscule": 0.5}},
        ],
        "demand_schedule": {0: 20, 5: 40},
    }

    with open(params_path, "w") as f:
        yaml.dump(config_data, f)

    seed = 42

    # Run 1
    result1 = run_single(str(params_path), seed=seed)

    # Run 2
    result2 = run_single(str(params_path), seed=seed)

    # --- Assertions ---

    # 1. Check graph structure
    assert result1.graph.number_of_nodes() == result2.graph.number_of_nodes()
    assert result1.graph.number_of_edges() == result2.graph.number_of_edges()
    assert list(result1.graph.nodes) == list(result2.graph.nodes)
    assert list(result1.graph.edges) == list(result2.graph.edges)

    # 2. Check node attributes
    for node in result1.graph.nodes:
        attrs1 = result1.graph.nodes[node]
        attrs2 = result2.graph.nodes[node]
        assert attrs1 == attrs2

    # 3. Check manuscript registry
    assert len(result1.state.registries.manuscripts) == len(result2.state.registries.manuscripts)
    for m_id, m1 in result1.state.registries.manuscripts.items():
        m2 = result2.state.registries.manuscripts.get(m_id)
        assert m1.manuscript_id == m2.manuscript_id
        assert m1.birth_tick == m2.birth_tick
        assert m1.death_tick == m2.death_tick
        assert m1.material == m2.material
        assert m1.region == m2.region
        assert m1.location == m2.location

    # 4. Check witness registry
    assert len(result1.state.registries.witnesses) == len(result2.state.registries.witnesses)
    for w_id, w1 in result1.state.registries.witnesses.items():
        w2 = result2.state.registries.witnesses.get(w_id)
        assert w1.witness_id == w2.witness_id
        assert w1.manuscript_id == w2.manuscript_id
        assert w1.script == w2.script

    # 5. Check instance texts for both regimes
    for regime in ["insertion", "omission"]:
        texts1 = result1.replays[regime].instance_texts
        texts2 = result2.replays[regime].instance_texts
        assert len(texts1) == len(texts2)
        for i_id, t1 in texts1.items():
            t2 = texts2[i_id]
            np.testing.assert_array_equal(t1, t2)

    # 6. Check alive manuscripts
    assert result1.state.alive_manuscripts == result2.state.alive_manuscripts

    # 7. Check telemetry
    assert len(result1.state.telemetry) == len(result2.state.telemetry)
    assert result1.state.telemetry == result2.state.telemetry


def test_different_seeds_produce_different_results(tmp_path):
    """
    Verify that two simulation runs with different seeds produce different results.
    """
    # Ensure experiments directory exists for run_single validation
    exp_dir = tmp_path / "experiments" / "diff_seeds_test"
    exp_dir.mkdir(parents=True)
    params_path = exp_dir / "params.yaml"

    config_data = {
        "total_ticks": 5,
        "text_length": 10,
        "p_region_migration": 0.5,
        "p_internal_relocation": 0.5,
        "reputation_distribution": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "pa_regime": "insertion",
        "pa_intervention_year": 1,
        "pa_intervention_region": "Asia Minor",
        "pa_innovator_reputation": 5.0,
        "material_transitions": [{"start_tick": 0, "distribution": {"parchment": 1.0}}],
        "script_transitions": [{"start_tick": 0, "distribution": {"uncial": 1.0}}],
        "demand_schedule": {0: 100},
    }
    with open(params_path, "w") as f:
        yaml.dump(config_data, f)

    result1 = run_single(str(params_path), seed=123)
    result2 = run_single(str(params_path), seed=456)

    # It's extremely unlikely (though theoretically possible) that two different seeds
    # would produce identical telemetry or alive manuscript sets in a stochastic simulation.
    assert result1.state.telemetry != result2.state.telemetry

    # Check that at least some physical properties differ
    # We compare the entire manuscript registries values
    m1_data = [m for m in result1.state.registries.manuscripts._manuscripts.values()]
    m2_data = [m for m in result2.state.registries.manuscripts._manuscripts.values()]

    # Locations should differ due to different seeds
    locs1 = [m.location for m in m1_data]
    locs2 = [m.location for m in m2_data]
    assert locs1 != locs2
