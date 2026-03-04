import numpy as np
import pytest

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_generator import extract_genealogy_snapshot, run_genealogy_generator
from pasim.core.rng import RNGContext
from pasim.core.text_replay import TextReplayEngine


@pytest.fixture
def config():
    return SimulationConfig(
        total_ticks=50,
        text_length=20,
        demand_schedule={1: 5, 10: 10, 25: 15},
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        material_transitions=[{"start_tick": 0, "distribution": {"parchment": 1.0}}],
        script_transitions=[{"start_tick": 0, "distribution": {"uncial": 1.0}}],
        pa_regime="insertion",
        pa_intervention_year=25,
        pa_intervention_region="Asia Minor",
        pa_innovator_reputation=5.0,
    )


def test_genealogy_determinism(config):
    """Verify that the same seed produces the exact same genealogy snapshot."""
    seed = 42

    rng1 = RNGContext(seed).spawn(1)[0]
    state1 = run_genealogy_generator(config.model_dump(), rng1)
    snapshot1 = extract_genealogy_snapshot(state1)

    rng2 = RNGContext(seed).spawn(1)[0]
    state2 = run_genealogy_generator(config.model_dump(), rng2)
    snapshot2 = extract_genealogy_snapshot(state2)

    assert len(snapshot1.nodes) == len(snapshot2.nodes)
    for n1, n2 in zip(snapshot1.nodes, snapshot2.nodes):
        assert n1.instance_id == n2.instance_id
        assert n1.parent_ids == n2.parent_ids
        assert n1.birth_tick == n2.birth_tick
        assert n1.region == n2.region
        assert n1.material == n2.material
        assert n1.reputation == n2.reputation


def test_text_replay_determinism(config):
    """Verify that replaying text on the same snapshot with the same seed is deterministic."""
    seed = 123
    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    replay_seed = 789
    engine1 = TextReplayEngine(config, snapshot, replay_seed)
    texts1 = engine1.run()

    engine2 = TextReplayEngine(config, snapshot, replay_seed)
    texts2 = engine2.run()

    assert texts1.keys() == texts2.keys()
    for tid in texts1:
        np.testing.assert_array_equal(texts1[tid], texts2[tid])


def test_regime_neutrality(config):
    """Verify that different regimes (insertion/omission) share the same parent graph but have different texts."""
    seed = 555
    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    # Create two configs with different regimes
    config_ins = config.model_copy(update={"pa_regime": "insertion"})
    config_omi = config.model_copy(update={"pa_regime": "omission"})

    replay_seed = 111
    engine_ins = TextReplayEngine(config_ins, snapshot, replay_seed)
    texts_ins = engine_ins.run()

    engine_omi = TextReplayEngine(config_omi, snapshot, replay_seed)
    texts_omi = engine_omi.run()

    assert texts_ins.keys() == texts_omi.keys()
    # Now they SHOULD differ starting from the root
    for tid in texts_ins:
        # Root is all 0s vs all 1s. This difference propagates.
        assert not np.array_equal(texts_ins[tid], texts_omi[tid])


def test_regime_dependent_autograph(config):
    """Verify that the autograph initialization is regime-dependent."""
    seed = 123
    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config.model_dump(), rng)
    snapshot = extract_genealogy_snapshot(state)

    # Check for "insertion" regime
    config_ins = config.model_copy(update={"pa_regime": "insertion"})
    engine_ins = TextReplayEngine(config_ins, snapshot, seed)
    texts_ins = engine_ins.run()
    # Find the autograph (root node with no parents)
    root_node = next(n for n in snapshot.nodes if not n.parent_ids)
    root_text_ins = texts_ins[root_node.instance_id]
    assert np.all(root_text_ins == 0)
    assert len(root_text_ins) == config.text_length

    # Check for "omission" regime
    config_omi = config.model_copy(update={"pa_regime": "omission"})
    engine_omi = TextReplayEngine(config_omi, snapshot, seed)
    texts_omi = engine_omi.run()
    root_text_omi = texts_omi[root_node.instance_id]
    assert np.all(root_text_omi == 1)
    assert len(root_text_omi) == config.text_length
