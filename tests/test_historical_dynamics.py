"""
These tests guarantee historical dynamics are deterministic, parameter-driven, and
isolated from core mechanics. They verify that scheduled events (like persecutions)
and environmental transitions (like material or script usage) behave correctly
and reproducibly.
"""

import copy
from collections import deque
from typing import Dict, List

import numpy as np
from pasim.config.schema import (
    DemandScheduleConfig,
    SimulationConfig,
    get_demand_for_tick,
)
from pasim.core.genealogy import add_root_node
from pasim.core.genealogy_generator import (
    _spawn_new_manuscripts_from_demand,
    handle_migration,
)
from pasim.core.historical_events import HistoricalEventManager
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
    death_tick: int = 100,  # Re-added
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

            manuscript = Manuscript(
                manuscript_id=manuscript_id,
                birth_tick=start_tick,
                death_tick=death_tick,
                material=Material.PARCHMENT,
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

            if state.graph.number_of_nodes() == 0:
                add_root_node(
                    graph=state.graph,
                    node_id=instance_id,
                    witness_id=witness_id,
                    manuscript_id=manuscript_id,
                    birth_tick=start_tick,
                    reputation=int(rng.integers(1, 6)),
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
                    reputation=int(rng.integers(1, 6)),
                )
                state.graph.add_edge(root_node_id, instance_id)  # Link to the root

            state.manuscript_to_instance_map[manuscript.manuscript_id] = instance_id

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
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        death_ticks=[],
        demand_schedule={0: {}},
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
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        death_ticks=[],
        demand_schedule={0: {}},
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
    # 1. Define a 2-stage material transition schedule
    material_schedule = [
        {
            "start_tick": 0,
            "distribution": {"papyrus": 1.0, "parchment": 0.0, "paper": 0.0},
        },
        {
            "start_tick": 5,
            "distribution": {"papyrus": 0.0, "parchment": 0.0, "paper": 1.0},
        },
    ]
    material_manager = MaterialTransitionManager(material_schedule)

    # Dummy script manager
    script_schedule = [{"start_tick": 0, "distribution": {"uncial": 1.0}}]
    script_manager = ScriptTransitionManager(script_schedule)

    dummy_config_data = {
        "total_ticks": 10,
        "p_region_migration": 0.0,
        "p_internal_relocation": 0.0,
        "reputation_distribution": {
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.2,
            5: 0.2,
        },  # Dummy 5-point distribution
        "death_ticks": [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
        ],  # Sufficient number
        "persecutions": [],
        "material_transitions": material_schedule,
        "script_transitions": script_schedule,
        "demand_schedule": {  # Use RootModel's direct structure
            0: {"Egypt": 2},
            6: {"Egypt": 3},
        },
    }
    dummy_simulation_config = SimulationConfig(**dummy_config_data)

    # 2. Setup simulation state
    rng_context = RNGContext(seed=42)
    rng = rng_context.spawn(1)[0]
    state = _create_initial_state(rng, {Region.EGYPT: 1}, dummy_simulation_config, state_collector_fixture)

    # Capture initial manuscript to check it doesn't change
    initial_manuscript_id = list(state.alive_manuscripts)[0]
    initial_manuscript = state.registries.manuscripts.get(initial_manuscript_id)
    assert initial_manuscript.material == Material.PARCHMENT

    # 3. Simulate spawning across the transition boundary
    test_death_ticks = deque(dummy_simulation_config.death_ticks)

    # Tick 2: Before transition (demand = 2, alive = 1 -> spawn 1)
    state.tick = 2
    demand_today_tick_2 = get_demand_for_tick(dummy_config_data, state.tick)
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_2,
        test_death_ticks,
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )

    # Tick 6: After transition (demand = 3, alive = 2 -> spawn 1)
    state.tick = 6
    demand_today_tick_6 = get_demand_for_tick(dummy_config_data, state.tick)
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_6,
        test_death_ticks,
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )

    # 4. Assertions
    manuscripts = list(state.registries.manuscripts.items())

    # Initial manuscript is unchanged
    assert manuscripts[0][1].material == Material.PARCHMENT

    # Manuscript spawned at tick 2 should be PAPYRUS
    m_tick2 = manuscripts[1][1]
    assert m_tick2.birth_tick == 2
    assert m_tick2.material == Material.PAPYRUS

    # Manuscript spawned at tick 6 should be PAPER
    m_tick6 = manuscripts[2][1]
    assert m_tick6.birth_tick == 6
    assert m_tick6.material == Material.PAPER


def test_script_transition(state_collector_fixture: List[GenerationState]):
    """Verify newly spawned witnesses use scripts based on the active schedule."""
    # 1. Define a 2-stage script transition schedule
    script_schedule = [
        {"start_tick": 0, "distribution": {"uncial": 1.0, "minuscule": 0.0}},
        {"start_tick": 5, "distribution": {"uncial": 0.0, "minuscule": 1.0}},
    ]
    script_manager = ScriptTransitionManager(script_schedule)

    # Dummy material manager
    material_schedule = [{"start_tick": 0, "distribution": {"parchment": 1.0}}]
    material_manager = MaterialTransitionManager(material_schedule)

    dummy_config_data = {
        "total_ticks": 10,
        "p_region_migration": 0.0,
        "p_internal_relocation": 0.0,
        "reputation_distribution": {
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.2,
            5: 0.2,
        },  # Dummy 5-point distribution
        "death_ticks": [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
        ],  # Sufficient number
        "persecutions": [],
        "material_transitions": material_schedule,
        "script_transitions": script_schedule,
        "demand_schedule": {  # Use RootModel's direct structure
            0: {"Egypt": 2},
            6: {"Egypt": 3},
        },
    }
    dummy_simulation_config = SimulationConfig(**dummy_config_data)

    # 2. Setup simulation state
    rng_context = RNGContext(seed=42)
    rng = rng_context.spawn(1)[0]
    state = _create_initial_state(rng, {Region.EGYPT: 1}, dummy_simulation_config, state_collector_fixture)

    initial_witness_id = list(state.registries.witnesses.items())[0][0]
    initial_witness = state.registries.witnesses.get(initial_witness_id)
    assert initial_witness.script == Script.UNCIAL

    # 3. Simulate spawning across the transition boundary
    test_death_ticks = deque(dummy_simulation_config.death_ticks)

    # Tick 2: Before transition
    state.tick = 2
    demand_today_tick_2 = get_demand_for_tick(dummy_config_data, state.tick)
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_2,
        test_death_ticks,
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )

    # Tick 6: After transition
    state.tick = 6
    demand_today_tick_6 = get_demand_for_tick(dummy_config_data, state.tick)
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_6,
        test_death_ticks,
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )

    # 4. Assertions
    witnesses = list(state.registries.witnesses.items())

    # Initial witness is unchanged
    assert witnesses[0][1].script == Script.UNCIAL

    # Witness spawned at tick 2 should be UNCIAL
    w_tick2 = witnesses[1][1]
    assert w_tick2.script == Script.UNCIAL

    # Witness spawned at tick 6 should be MINUSCULE
    w_tick6 = witnesses[2][1]
    assert w_tick6.script == Script.MINUSCULE


def test_demand_schedule(state_collector_fixture: List[GenerationState]):
    """Verify demand schedule correctly drives spawning and handles missing ticks."""
    # 1. Define demand schedule data and managers
    demand_schedule_data = {
        0: {"Egypt": 5, "Asia Minor": 3},
        10: {"Egypt": 8},
    }
    material_schedule_config = [{"start_tick": 0, "distribution": {"parchment": 1.0}}]
    material_manager = MaterialTransitionManager(material_schedule_config)
    script_schedule_config = [{"start_tick": 0, "distribution": {"uncial": 1.0}}]
    script_manager = ScriptTransitionManager(script_schedule_config)

    # Create a full dummy parameters dictionary that would pass SimulationConfig validation
    dummy_full_params = {
        "total_ticks": 20,
        "p_region_migration": 0.0,
        "p_internal_relocation": 0.0,
        "reputation_distribution": {
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.2,
            5: 0.2,
        },  # Example 5-point distribution
        "death_ticks": [100] * 20,  # Example death ticks
        "persecutions": [],
        "material_transitions": material_schedule_config,
        "script_transitions": script_schedule_config,
        "demand_schedule": demand_schedule_data,
    }
    dummy_simulation_config = SimulationConfig(**dummy_full_params)

    # 2. Test last-known-value retrieval
    assert get_demand_for_tick(dummy_full_params, 0) == {
        Region.EGYPT: 5,
        Region.ASIA_MINOR: 3,
    }
    assert get_demand_for_tick(dummy_full_params, 5) == {
        Region.EGYPT: 5,
        Region.ASIA_MINOR: 3,
    }  # Falls back to tick 0
    assert get_demand_for_tick(dummy_full_params, 10) == {Region.EGYPT: 8}
    assert get_demand_for_tick(dummy_full_params, 15) == {Region.EGYPT: 8}  # Falls back to tick 10

    # 3. Test spawning to meet demand
    rng_context = RNGContext(seed=1)
    rng = rng_context.spawn(1)[0]
    state = _create_initial_state(
        rng, {Region.EGYPT: 2, Region.ASIA_MINOR: 1}, dummy_simulation_config, state_collector_fixture
    )  # Start below demand

    # Run spawning at tick 5 (should use demand from tick 0)
    state.tick = 5

    demand_today_tick_5 = get_demand_for_tick(dummy_full_params, state.tick)
    test_death_ticks_spawn = deque([100] * 10)  # Enough death ticks for spawning test
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_5,
        test_death_ticks_spawn,
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )

    # Assertions for tick 5
    alive_egypt = [
        state.registries.manuscripts.get(mid)
        for mid in state.alive_manuscripts
        if state.registries.manuscripts.get(mid).region == Region.EGYPT
    ]
    alive_asia_minor = [
        state.registries.manuscripts.get(mid)
        for mid in state.alive_manuscripts
        if state.registries.manuscripts.get(mid).region == Region.ASIA_MINOR
    ]
    assert len(alive_egypt) == 5  # 2 initial + 3 spawned
    assert len(alive_asia_minor) == 3  # 1 initial + 2 spawned

    # Run spawning at tick 12 (should use demand from tick 10)
    state.tick = 12
    demand_today_tick_12 = get_demand_for_tick(dummy_full_params, state.tick)
    _spawn_new_manuscripts_from_demand(
        state,
        demand_today_tick_12,
        test_death_ticks_spawn,  # Reuse deque or create new if needed
        dummy_simulation_config,
        rng,
        material_manager,
        script_manager,
    )

    # Assertions for tick 12
    alive_egypt = [
        state.registries.manuscripts.get(mid)
        for mid in state.alive_manuscripts
        if state.registries.manuscripts.get(mid).region == Region.EGYPT
    ]
    alive_asia_minor = [
        state.registries.manuscripts.get(mid)
        for mid in state.alive_manuscripts
        if state.registries.manuscripts.get(mid).region == Region.ASIA_MINOR
    ]
    assert len(alive_egypt) == 8  # 5 existing + 3 spawned
    assert len(alive_asia_minor) == 3  # No new demand for Asia Minor, so no change


def test_migration_determinism(state_collector_fixture: List[GenerationState]):
    """Verify manuscript migration is deterministic for a given seed."""

    def run_migration_sim(seed: int, state_collector: List[GenerationState]) -> Dict[int, Dict[str, Region]]:
        """Run a few ticks of migration and return the history."""
        dummy_config = SimulationConfig(
            total_ticks=1,
            text_length=100,
            p_region_migration=0.0,
            p_internal_relocation=0.0,
            reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
            death_ticks=[],
            demand_schedule={0: {}},
        )
        rng_context = RNGContext(seed=seed)
        rng = rng_context.spawn(1)[0]
        state = _create_initial_state(rng, {Region.EGYPT: 10, Region.ASIA_MINOR: 10}, dummy_config, state_collector)

        history = {}
        for tick in range(1, 5):
            state.tick = tick
            handle_migration(state, rng, p_region_migration=0.5, p_internal_relocation=0.5)

            # Record the region of each manuscript at this tick
            history[tick] = {m: state.registries.manuscripts.get(m).region for m in state.alive_manuscripts}
        return history

    # Run simulation twice with the same seed
    history1 = run_migration_sim(seed=999, state_collector=state_collector_fixture)
    history2 = run_migration_sim(seed=999, state_collector=state_collector_fixture)

    # Assert histories are identical
    assert history1 == history2

    # Run with a different seed
    history3 = run_migration_sim(seed=111, state_collector=state_collector_fixture)

    # Assert history is different (it's probabilistically unlikely to be the same)
    assert history1 != history3


def test_event_ordering_stability(state_collector_fixture: List[GenerationState]):
    """Ensure event application is stable regardless of config order."""
    event1 = {
        "event_type": "persecution",
        "start_tick": 5,
        "regions": ["EGYPT"],
        "kill_proportion": 0.2,
    }
    event2 = {
        "event_type": "persecution",
        "start_tick": 2,
        "regions": ["EGYPT"],
        "kill_proportion": 0.5,
    }

    dummy_config = SimulationConfig(
        total_ticks=1,
        text_length=100,
        p_region_migration=0.0,
        p_internal_relocation=0.0,
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        death_ticks=[],
        demand_schedule=DemandScheduleConfig(root={0: {}}),
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

    # Run with both orderings using the same seed
    destroyed_ids1 = run_with_order([event1, event2], seed=77, state_collector=state_collector_fixture)
    destroyed_ids2 = run_with_order([event2, event1], seed=77, state_collector=state_collector_fixture)

    # The set of destroyed manuscripts should be identical
    assert destroyed_ids1 == destroyed_ids2
    dummy_config = SimulationConfig(
        total_ticks=1,
        text_length=100,
        p_region_migration=0.0,
        p_internal_relocation=0.0,
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        death_ticks=[],
        demand_schedule=DemandScheduleConfig(root={0: {}}),
    )
    rng_context = RNGContext(seed=1)
    rng = rng_context.spawn(1)[0]
    state = _create_initial_state(rng, {Region.EGYPT: 10}, dummy_config, state_collector_fixture)
    state.tick = 5

    # 1. Test HistoricalEventManager
    original_tick = state.tick
    original_alive_count = len(state.alive_manuscripts)
    event_manager = HistoricalEventManager([])  # No events
    event_manager.apply_events_for_tick(state, rng)
    assert state.tick == original_tick
    assert len(state.alive_manuscripts) == original_alive_count

    # 2. Test MaterialTransitionManager
    material_manager = MaterialTransitionManager([{"start_tick": 0, "distribution": {"parchment": 1.0}}])
    state_before = copy.deepcopy(state)
    _ = material_manager.get_material_for_tick(1, rng)
    assert state.tick == state_before.tick
    assert state.alive_manuscripts == state_before.alive_manuscripts

    # 3. Test ScriptTransitionManager
    script_manager = ScriptTransitionManager([{"start_tick": 0, "distribution": {"uncial": 1.0}}])
    state_before = copy.deepcopy(state)
    _ = script_manager.get_script_for_tick(1, rng)
    assert state.tick == state_before.tick
    assert state.alive_manuscripts == state_before.alive_manuscripts
