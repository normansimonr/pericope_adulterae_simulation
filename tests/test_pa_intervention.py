from unittest.mock import patch

import numpy as np
import pytest

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_generator import extract_genealogy_snapshot, run_genealogy_generator
from pasim.core.rng import RNGContext
from pasim.core.scribal_rules import apply_scribal_rule
from pasim.core.text_replay import TextReplayEngine


@pytest.fixture
def base_config_data():
    return {
        "total_ticks": 50,
        "text_length": 20,
        "demand_schedule": {1: 5, 25: 10, 50: 10},  # 5 born at 1, 5 born at 25
        "reputation_distribution": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "pa_regime": "insertion",
        "pa_intervention_year": 25,
        "pa_intervention_region": "Asia Minor",
        "pa_innovator_reputation": 5.0,
        "material_transitions": [{"start_tick": 0, "distribution": {"parchment": 1.0}}],
        "script_transitions": [{"start_tick": 0, "distribution": {"uncial": 1.0}}],
    }


def test_pa_intervention_insertion_regime(base_config_data):
    """Verify intervention in insertion regime: autograph 0s, innovator 1s."""
    seed = 42
    config = SimulationConfig(**base_config_data)

    # Run demographic simulation
    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    # Run text replay
    engine = TextReplayEngine(config, snapshot, seed)
    texts = engine.run()

    # Check autograph (all 0s)
    root_node = next(n for n in snapshot.nodes if not n.parent_ids)
    assert np.all(texts[root_node.instance_id] == 0)

    # Check innovator (all 1s)
    innovator_id = engine.innovator_id
    assert innovator_id is not None
    assert np.all(texts[innovator_id] == 1)

    # Check innovator attributes
    innovator_node = next(n for n in snapshot.nodes if n.instance_id == innovator_id)
    assert innovator_node.birth_tick == config.pa_intervention_year
    assert innovator_node.region == config.pa_intervention_region


def test_pa_intervention_omission_regime(base_config_data):
    """Verify intervention in omission regime: autograph 1s, innovator 0s."""
    base_config_data["pa_regime"] = "omission"
    seed = 42
    config = SimulationConfig(**base_config_data)

    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    engine = TextReplayEngine(config, snapshot, seed)
    texts = engine.run()

    # Check autograph (all 1s)
    root_node = next(n for n in snapshot.nodes if not n.parent_ids)
    assert np.all(texts[root_node.instance_id] == 1)

    # Check innovator (all 0s)
    innovator_id = engine.innovator_id
    assert np.all(texts[innovator_id] == 0)


def test_pa_intervention_no_eligible_nodes(base_config_data):
    """Verify error if no nodes are born in the intervention year/region."""
    # Set intervention year where no demand exists (though demand fallback might spawn some)
    # Let's set a region that has 0 distribution if possible.
    # Asia Minor has 1.0 distribution in later centuries, but Egypt has 0.0.
    base_config_data["total_ticks"] = 1000
    base_config_data["pa_intervention_year"] = 900
    base_config_data["pa_intervention_region"] = "Egypt"  # 0 demand in century 6+

    config = SimulationConfig(**base_config_data)

    rng = RNGContext(seed=123).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    with pytest.raises(RuntimeError) as excinfo:
        TextReplayEngine(config, snapshot, seed=123)
    assert "No eligible nodes found" in str(excinfo.value)


def test_pa_intervention_determinism(base_config_data):
    """Verify same seed selects same innovator."""
    seed = 999
    config = SimulationConfig(**base_config_data)

    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    engine1 = TextReplayEngine(config, snapshot, seed)
    engine2 = TextReplayEngine(config, snapshot, seed)

    assert engine1.innovator_id == engine2.innovator_id
    assert engine1.innovator_id is not None


def test_pa_intervention_reputation_override(base_config_data):
    """Verify reputation override for innovator (using mock to intercept call)."""
    seed = 42
    config = SimulationConfig(**base_config_data)

    # Run demographic simulation
    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    # Patch apply_scribal_rule to check reputation passed
    with patch("pasim.core.text_replay.apply_scribal_rule", wraps=apply_scribal_rule) as mock_rule:
        engine = TextReplayEngine(config, snapshot, seed)
        engine.run()

        # Find the call for the innovator
        # Note: apply_scribal_rule is only called if node has parents.
        # Our innovator at year 25 should have parents.
        found_innovator_call = False
        for call in mock_rule.call_args_list:
            if call.kwargs.get("reputation") == int(config.pa_innovator_reputation):
                found_innovator_call = True

        assert found_innovator_call


def test_pa_intervention_integrity(base_config_data):
    """Verify that text replay does NOT modify the genealogy graph structure."""
    seed = 42
    config = SimulationConfig(**base_config_data)

    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    # Record graph state before replay
    nodes_before = list(state.graph.nodes())
    edges_before = list(state.graph.edges())
    attrs_before = {n: state.graph.nodes[n].copy() for n in nodes_before}

    # Run replay
    engine = TextReplayEngine(config, snapshot, seed)
    engine.run()

    # Assert graph unchanged
    assert list(state.graph.nodes()) == nodes_before
    assert list(state.graph.edges()) == edges_before
    for n in nodes_before:
        assert state.graph.nodes[n] == attrs_before[n]


def test_pa_intervention_exactly_once(base_config_data):
    """Verify that intervention is applied exactly once."""
    seed = 42
    config = SimulationConfig(**base_config_data)

    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    engine = TextReplayEngine(config, snapshot, seed)

    # Mock _process_node to count applications
    original_process_node = engine._process_node
    applications = []

    def wrapped_process_node(node, rng):
        is_innovator = node.instance_id == engine.innovator_id
        if is_innovator:
            applications.append(node.instance_id)
        return original_process_node(node, rng)

    engine._process_node = wrapped_process_node
    engine.run()

    assert len(applications) == 1
    assert applications[0] == engine.innovator_id
