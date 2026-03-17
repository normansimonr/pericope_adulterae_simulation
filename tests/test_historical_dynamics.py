"""
These tests guarantee historical dynamics are deterministic, parameter-driven, and
isolated from core mechanics. They verify that scheduled events (like persecutions)
and environmental transitions (like material or script usage) behave correctly
and reproducibly.
"""

import copy
from typing import Any, Dict, List, cast

import numpy as np

from pasim.config.schema import (
    DemandScheduleConfig,
    MaterialTransitionConfig,
    ScriptTransitionConfig,
    SimulationConfig,
)
from pasim.core.genealogy import add_root_node
from pasim.core.genealogy_generator import (
    _allocate_demand,
    _spawn_new_manuscripts_from_demand,
    handle_migration,
)
from pasim.core.historical_events import HistoricalEventManager
from pasim.core.lifespan import sample_lifespan
from pasim.core.material_transition_manager import MaterialTransitionManager
from pasim.core.rng import RNGContext
from pasim.core.script_transition_manager import ScriptTransitionManager
from pasim.core.simulation_state import GenerationState, initialise_generation_state
from pasim.core.state import Manuscript, Material, Region, Script, Witness
from pasim.core.text_initialisation import make_initial_text


def _create_initial_state(
    rng: np.random.Generator,
    manuscript_counts: Dict[Region, int],
    config: SimulationConfig,
    state_collector_fixture: List[GenerationState],
    start_tick: int = 0,
) -> GenerationState:
    """Helper to create a generation state with a set number of manuscripts in specific regions."""
    state = initialise_generation_state()
    state_collector_fixture.append(state)
    state.tick = start_tick

    for region, count in manuscript_counts.items():
        for i in range(count):
            manuscript_id = f"M_{region.name}_{i}"
            witness_id = f"W_{region.name}_{i}"
            instance_id = f"I_{region.name}_{i}"

            # Assume PARCHMENT for initial state for simplicity, as it was before.
            # Lifespan is now sampled based on material and region.
            material = Material.PARCHMENT
            lifespan = sample_lifespan(material=material, region=region, rng=rng, config=config)
            death_tick_for_initial_manuscript = start_tick + lifespan

            manuscript = Manuscript(
                manuscript_id=manuscript_id,
                birth_tick=start_tick,
                death_tick=death_tick_for_initial_manuscript,
                material=material,
                region=region,
                location=(rng.random(), rng.random()),
            )
            witness = Witness(
                witness_id=witness_id,
                manuscript_id=manuscript_id,
                script=Script.UNCIAL,
            )

            state.registries.manuscripts.add(manuscript)
            state.registries.witnesses.add(witness)
            state.alive_manuscripts.add(manuscript.manuscript_id)
            state.alive_by_region[region].add(manuscript.manuscript_id)

            reputation = int(rng.integers(1, 6))

            if state.graph.number_of_nodes() == 0:
                add_root_node(
                    graph=state.graph,
                    node_id=instance_id,
                    witness_id=witness_id,
                    manuscript_id=manuscript_id,
                    birth_tick=start_tick,
                    reputation=reputation,
                )
            else:
                # For subsequent manuscripts, add as child nodes of the initial root
                # This ensures graph validity when creating multiple initial manuscripts
                # without violating the single root invariant.
                root_node_id = list(state.graph.nodes)[0]  # Get the ID of the first (root) node
                state.graph.add_node(
                    instance_id,
                    witness_id=witness_id,
                    manuscript_id=manuscript_id,
                    birth_tick=start_tick,
                    reputation=reputation,
                )
                state.graph.add_edge(root_node_id, instance_id)  # Link to the root

            state.manuscript_to_instance_map[manuscript.manuscript_id] = instance_id
            state.instance_reputations[instance_id] = reputation

            # Create and store initial text for the new instance
            initial_text = make_initial_text(config)
            state.registries.instance_texts[instance_id] = initial_text

    return state


def test_persecution_correctness(state_collector_fixture: List[GenerationState]):
    """Verify persecution event removes correct proportion of manuscripts from a targeted region."""
    rng_context = RNGContext(seed=123)
    rng = rng_context.spawn(1)[0]

    # Define a dummy config for _create_initial_state
    dummy_config = SimulationConfig(
        total_ticks=1,
        text_length=100,
        p_region_migration=0.0,
        p_internal_relocation=0.0,
        p_uncial_exemplar_death_on_minuscule_birth=0.0,
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        geographical_candidate_pool_size=10,
        pa_intervention_radius=10.0,
        demand_schedule=DemandScheduleConfig({0: 0}),
        pa_regime="insertion",
        pa_intervention_year=0,
        pa_intervention_region=Region.ASIA_MINOR,
        pa_innovator_reputation=5.0,
        log_tick_frequency=100,
        validation_frequency=0,
    )

    # 1. Initialize state with manuscripts in two regions
    initial_counts = {Region.EGYPT: 10, Region.ASIA_MINOR: 10}
    state = _create_initial_state(rng, initial_counts, dummy_config, state_collector_fixture)

    total_manuscripts = sum(initial_counts.values())
    assert len(state.alive_manuscripts) == total_manuscripts
    assert state.graph.number_of_nodes() == total_manuscripts

    # 2. Apply a persecution event
    event_config = {
        "event_type": "persecution",
        "start_tick": 1,
        "end_tick": 1,
        "regions": [Region.EGYPT.value],
        "kill_proportion": 0.5,
    }
    event_manager = HistoricalEventManager([event_config])

    state.tick = 1
    event_manager.apply_events_for_tick(state, rng)

    # 3. Assertions
    # 3.1. Approximately half of Egypt's manuscripts are destroyed
    egypt_survivors = [
        state.registries.manuscripts.get(m) for m in state.alive_manuscripts if state.registries.manuscripts.get(m).region == Region.EGYPT
    ]
    assert len(egypt_survivors) == 5

    # 3.2. Asia Minor's manuscripts are untouched
    asia_minor_survivors = [
        state.registries.manuscripts.get(m)
        for m in state.alive_manuscripts
        if state.registries.manuscripts.get(m).region == Region.ASIA_MINOR
    ]
    assert len(asia_minor_survivors) == 10

    # 3.3. Total alive manuscripts is reduced as expected
    assert len(state.alive_manuscripts) == 15

    # 3.4. No genealogy nodes were deleted
    assert state.graph.number_of_nodes() == total_manuscripts


def test_persecution_determinism(state_collector_fixture: List[GenerationState]):
    """Verify persecution events are deterministic with the same seed and different otherwise."""
    initial_counts = {Region.EGYPT: 50}
    event_config = {
        "event_type": "persecution",
        "start_tick": 1,
        "end_tick": 1,
        "regions": [Region.EGYPT.value],
        "kill_proportion": 0.5,
    }
    event_manager = HistoricalEventManager([event_config])

    dummy_config = SimulationConfig(
        total_ticks=1,
        text_length=100,
        p_region_migration=0.0,
        p_internal_relocation=0.0,
        p_uncial_exemplar_death_on_minuscule_birth=0.0,
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        geographical_candidate_pool_size=10,
        pa_intervention_radius=10.0,
        demand_schedule=DemandScheduleConfig({0: 0}),
        pa_regime="insertion",
        pa_intervention_year=0,
        pa_intervention_region=Region.ASIA_MINOR,
        pa_innovator_reputation=5.0,
        log_tick_frequency=100,
        validation_frequency=0,
    )

    # --- Run 1 with seed 123 ---
    rng_context1 = RNGContext(seed=123)
    rng1 = rng_context1.spawn(1)[0]
    state1 = _create_initial_state(rng1, initial_counts, dummy_config, state_collector_fixture)
    initial_ids = {m for m in state1.alive_manuscripts}

    state1.tick = 1
    event_manager.apply_events_for_tick(state1, rng1)
    destroyed_ids1 = initial_ids - {m for m in state1.alive_manuscripts}

    # --- Run 2 with seed 123 ---
    rng_context2 = RNGContext(seed=123)
    rng2 = rng_context2.spawn(1)[0]
    state2 = _create_initial_state(rng2, initial_counts, dummy_config, state_collector_fixture)

    state2.tick = 1
    event_manager.apply_events_for_tick(state2, rng2)
    destroyed_ids2 = initial_ids - {m for m in state2.alive_manuscripts}

    # --- Assert identical results for same seed ---
    assert destroyed_ids1 == destroyed_ids2

    # --- Run 3 with seed 456 ---
    rng_context3 = RNGContext(seed=456)
    rng3 = rng_context3.spawn(1)[0]
    state3 = _create_initial_state(rng3, initial_counts, dummy_config, state_collector_fixture)

    state3.tick = 1
    event_manager.apply_events_for_tick(state3, rng3)
    destroyed_ids3 = initial_ids - {m for m in state3.alive_manuscripts}

    # --- Assert different results for different seed ---
    assert destroyed_ids1 != destroyed_ids3


def test_material_transition(state_collector_fixture: List[GenerationState]):
    """Verify newly spawned manuscripts use materials based on the active schedule."""
    material_schedule = [
        {"start_tick": 0, "distribution": {"papyrus": 1.0}},
        {"start_tick": 5, "distribution": {"parchment": 1.0}},
    ]
    material_manager = MaterialTransitionManager(material_schedule)

    script_schedule = [{"start_tick": 0, "distribution": {"uncial": 1.0}}]
    script_manager = ScriptTransitionManager(script_schedule)

    dummy_simulation_config = SimulationConfig(
        total_ticks=10,
        text_length=100,
        p_region_migration=0.0,
        p_internal_relocation=0.0,
        p_uncial_exemplar_death_on_minuscule_birth=0.0,
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        geographical_candidate_pool_size=10,
        pa_intervention_radius=10.0,
        pa_regime="insertion",
        pa_intervention_year=5,
        pa_intervention_region=Region.ASIA_MINOR,
        pa_innovator_reputation=5.0,
        persecutions=[],
        material_transitions=[MaterialTransitionConfig(**cast(Any, m)) for m in material_schedule],
        script_transitions=[ScriptTransitionConfig(**cast(Any, s)) for s in script_schedule],
        demand_schedule=DemandScheduleConfig({0: 2, 6: 3}),
        log_tick_frequency=100,
        validation_frequency=0,
    )

    rng_context = RNGContext(seed=42)
    rng = rng_context.spawn(1)[0]
    state = initialise_generation_state()
    state_collector_fixture.append(state)
    state.tick = 0

    state.tick = 2
    aggregate_demand_tick_2 = dummy_simulation_config.demand_schedule.root.get(0, 0)
    demand_today_tick_2 = _allocate_demand(state.tick, aggregate_demand_tick_2, dummy_simulation_config)
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_2,
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )
    assert len(state.alive_manuscripts) == 4

    state.tick = 6
    aggregate_demand_tick_6 = dummy_simulation_config.demand_schedule.root.get(6, 0)
    demand_today_tick_6 = _allocate_demand(state.tick, aggregate_demand_tick_6, dummy_simulation_config)
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_6,
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )
    assert len(state.alive_manuscripts) == 5

    m_tick2 = next((m for m in state.registries.manuscripts._manuscripts.values() if m.birth_tick == 2), None)
    assert m_tick2 is not None
    assert m_tick2.material == Material.PAPYRUS

    m_tick6 = next((m for m in state.registries.manuscripts._manuscripts.values() if m.birth_tick == 6), None)
    assert m_tick6 is not None
    assert m_tick6.material == Material.PARCHMENT


def test_script_transition(state_collector_fixture: List[GenerationState]):
    """Verify newly spawned witnesses use scripts based on the active schedule."""
    script_schedule = [
        {"start_tick": 0, "distribution": {"uncial": 1.0, "minuscule": 0.0}},
        {"start_tick": 5, "distribution": {"uncial": 0.0, "minuscule": 1.0}},
    ]
    script_manager = ScriptTransitionManager(script_schedule)

    material_schedule = [{"start_tick": 0, "distribution": {"parchment": 1.0}}]
    material_manager = MaterialTransitionManager(material_schedule)

    dummy_simulation_config = SimulationConfig(
        total_ticks=10,
        text_length=100,
        p_region_migration=0.0,
        p_internal_relocation=0.0,
        p_uncial_exemplar_death_on_minuscule_birth=0.0,
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        geographical_candidate_pool_size=10,
        pa_intervention_radius=10.0,
        pa_regime="insertion",
        pa_intervention_year=5,
        pa_intervention_region=Region.ASIA_MINOR,
        pa_innovator_reputation=5.0,
        persecutions=[],
        material_transitions=[MaterialTransitionConfig(**cast(Any, m)) for m in material_schedule],
        script_transitions=[ScriptTransitionConfig(**cast(Any, s)) for s in script_schedule],
        demand_schedule=DemandScheduleConfig({0: 2, 6: 3}),
        log_tick_frequency=100,
        validation_frequency=0,
    )

    rng_context = RNGContext(seed=42)
    rng = rng_context.spawn(1)[0]
    state = initialise_generation_state()
    state_collector_fixture.append(state)
    state.tick = 0

    state.tick = 2
    aggregate_demand_tick_2 = dummy_simulation_config.demand_schedule.root.get(0, 0)
    demand_today_tick_2 = _allocate_demand(state.tick, aggregate_demand_tick_2, dummy_simulation_config)
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_2,
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )
    assert len(state.alive_manuscripts) == 4

    state.tick = 6
    aggregate_demand_tick_6 = dummy_simulation_config.demand_schedule.root.get(6, 0)
    demand_today_tick_6 = _allocate_demand(state.tick, aggregate_demand_tick_6, dummy_simulation_config)
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_6,
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )
    assert len(state.alive_manuscripts) == 5

    m_ids_tick2 = [m.manuscript_id for m in state.registries.manuscripts._manuscripts.values() if m.birth_tick == 2]
    w_tick2 = next((w for w in state.registries.witnesses._witnesses.values() if w.manuscript_id in m_ids_tick2), None)
    assert w_tick2 is not None
    assert w_tick2.script == Script.UNCIAL

    m_ids_tick6 = [m.manuscript_id for m in state.registries.manuscripts._manuscripts.values() if m.birth_tick == 6]
    w_tick6 = next((w for w in state.registries.witnesses._witnesses.values() if w.manuscript_id in m_ids_tick6), None)
    assert w_tick6 is not None
    assert w_tick6.script == Script.MINUSCULE


def test_migration_determinism(state_collector_fixture: List[GenerationState]):
    """Verify manuscript migration is deterministic for a given seed."""

    def run_migration_sim(seed: int, state_collector: List[GenerationState]) -> Dict[int, Dict[str, Region]]:
        dummy_config = SimulationConfig(
            total_ticks=10,
            text_length=100,
            p_region_migration=0.5,
            p_internal_relocation=0.5,
            p_uncial_exemplar_death_on_minuscule_birth=0.0,
            reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
            geographical_candidate_pool_size=10,
            pa_intervention_radius=10.0,
            demand_schedule=DemandScheduleConfig({0: 0}),
            pa_regime="insertion",
            pa_intervention_year=0,
            pa_intervention_region=Region.ASIA_MINOR,
            pa_innovator_reputation=5.0,
            log_tick_frequency=100,
            validation_frequency=0,
        )
        rng_context = RNGContext(seed=seed)
        rng = rng_context.spawn(1)[0]
        state = _create_initial_state(rng, {Region.EGYPT: 10, Region.ASIA_MINOR: 10}, dummy_config, state_collector)

        history = {}
        for tick in range(1, 5):
            state.tick = tick
            handle_migration(state, rng, dummy_config)
            history[tick] = {m: state.registries.manuscripts.get(m).region for m in state.alive_manuscripts}
        return history

    history1 = run_migration_sim(seed=999, state_collector=state_collector_fixture)
    history2 = run_migration_sim(seed=999, state_collector=state_collector_fixture)
    assert history1 == history2

    history3 = run_migration_sim(seed=111, state_collector=state_collector_fixture)
    assert history1 != history3


def test_event_ordering_stability(state_collector_fixture: List[GenerationState]):
    """Ensure event application is stable regardless of config order."""
    event1 = {"event_type": "persecution", "start_tick": 5, "regions": ["Egypt"], "kill_proportion": 0.2}
    event2 = {"event_type": "persecution", "start_tick": 2, "regions": ["Egypt"], "kill_proportion": 0.5}

    dummy_config = SimulationConfig(
        total_ticks=10,
        text_length=100,
        p_region_migration=0.0,
        p_internal_relocation=0.0,
        p_uncial_exemplar_death_on_minuscule_birth=0.0,
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        geographical_candidate_pool_size=10,
        pa_intervention_radius=10.0,
        demand_schedule=DemandScheduleConfig({0: 0}),
        pa_regime="insertion",
        pa_intervention_year=0,
        pa_intervention_region=Region.ASIA_MINOR,
        pa_innovator_reputation=5.0,
        log_tick_frequency=100,
        validation_frequency=0,
    )

    def run_with_order(order: list, seed: int, state_collector: List[GenerationState]) -> set:
        rng_context = RNGContext(seed=seed)
        rng = rng_context.spawn(1)[0]
        state = _create_initial_state(rng, {Region.EGYPT: 100}, dummy_config, state_collector)
        initial_ids = {m for m in state.alive_manuscripts}

        event_manager = HistoricalEventManager(order)
        for i in range(1, 7):
            state.tick = i
            event_manager.apply_events_for_tick(state, rng)
        return initial_ids - {m for m in state.alive_manuscripts}

    destroyed_ids1 = run_with_order([event1, event2], 77, state_collector_fixture)
    destroyed_ids2 = run_with_order([event2, event1], 77, state_collector_fixture)
    assert destroyed_ids1 == destroyed_ids2


def test_manager_independence(state_collector_fixture: List[GenerationState]):
    """Verify that managers do not interfere with each other or the simulation state."""
    dummy_config = SimulationConfig(
        total_ticks=10,
        text_length=100,
        p_region_migration=0.0,
        p_internal_relocation=0.0,
        p_uncial_exemplar_death_on_minuscule_birth=0.0,
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        geographical_candidate_pool_size=10,
        pa_intervention_radius=10.0,
        demand_schedule=DemandScheduleConfig({0: 0}),
        pa_regime="insertion",
        pa_intervention_year=0,
        pa_intervention_region=Region.ASIA_MINOR,
        pa_innovator_reputation=5.0,
        log_tick_frequency=100,
        validation_frequency=0,
    )
    rng_context = RNGContext(seed=1)
    rng = rng_context.spawn(1)[0]
    state = _create_initial_state(rng, {Region.EGYPT: 10}, dummy_config, state_collector_fixture)
    state.tick = 5

    # 1. Test HistoricalEventManager
    original_tick = state.tick
    original_alive_count = len(state.alive_manuscripts)
    event_manager = HistoricalEventManager([])
    event_manager.apply_events_for_tick(state, rng)
    assert state.tick == original_tick
    assert len(state.alive_manuscripts) == original_alive_count

    # 2. Test MaterialTransitionManager
    material_manager = MaterialTransitionManager([{"start_tick": 0, "distribution": {"parchment": 1.0}}])
    state_before = copy.deepcopy(state)
    _ = material_manager.get_active_distribution(1)
    assert state.tick == state_before.tick
    assert state.alive_manuscripts == state_before.alive_manuscripts

    # 3. Test ScriptTransitionManager
    script_manager = ScriptTransitionManager([{"start_tick": 0, "distribution": {"uncial": 1.0}}])
    state_before = copy.deepcopy(state)
    _ = script_manager.get_active_distribution(1)
    assert state.tick == state_before.tick
    assert state.alive_manuscripts == state_before.alive_manuscripts
