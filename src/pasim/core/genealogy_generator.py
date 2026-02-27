"""
Genealogy Generator with Death, Migration, and Demand-Based Spawning.

This module provides a tick-based orchestration for generating a manuscript
genealogy. It defines the core control flow, state representation, and interfaces
for the generation process.

Temporal Execution Pipeline:
The simulation advances tick by tick, with each tick comprising a series of
discrete stages executed in a strict order to ensure determinism and logical
consistency. The pipeline is as follows:
1.  **Advance Tick**: The global simulation clock is incremented.
2.  **Handle Scheduled Deaths**: Manuscripts scheduled to "die" on the current
    tick are removed from the pool of living manuscripts available for copying.
3.  **Apply Historical Events**: Exogenous shocks (e.g., persecutions) managed
    by the HistoricalEventManager are applied, potentially altering the state
    of existing manuscripts.
4.  **Handle Migration**: Living manuscripts may relocate between or within
    regions, affecting their geographical properties.
5.  **Spawn New Manuscripts**: New manuscripts are created to meet regional demand,
    with their properties (e.g., material, script) determined by the active
    environmental transition regimes.
6.  **Select Exemplars**: For each newly spawned manuscript, parent exemplars
    are selected from the pool of living, post-migration manuscripts.
7.  **Scribal Copying**: The textual content of new manuscripts is generated
    based on their selected exemplars, including any scribal alterations, and
    stored in the state.

The generator's design enforces a strict separation of concerns:
- **Genealogy Graph**: Nodes are abstract "witness instances." The graph topology
  (who copied from whom) is its sole concern.
- **Manuscript Registry**: Holds the rich metadata for physical manuscripts,
  such as region, material, and birth/death ticks.

This separation ensures that attributes of the physical artifacts do not pollute
the abstract genealogical structure.

Explicit Exclusions:
- Exemplar selection policies.
- Contamination and scribal error models.
- Batch execution or file I/O.
"""

import logging
import time
from math import ceil
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.random import Generator as RNG
from scipy.spatial import KDTree

from pasim.config.schema import SimulationConfig
from pasim.core.exemplar_selection import select_exemplars
from pasim.core.genealogy import add_child_node, add_root_node
from pasim.core.historical_events import HistoricalEventManager
from pasim.core.lifespan import sample_lifespan
from pasim.core.material_transition_manager import MaterialTransitionManager
from pasim.core.reputation import sample_reputation
from pasim.core.scribal_rules import apply_scribal_rule
from pasim.core.script_transition_manager import ScriptTransitionManager
from pasim.core.simulation_state import GenerationState, initialise_generation_state
from pasim.core.spatial import generate_random_coordinates
from pasim.core.state import DeathReason, Manuscript, Region, Witness
from pasim.core.text_initialisation import make_initial_text

logger = logging.getLogger(__name__)


# --- Regional Demand Allocation ---
REGIONAL_DISTRIBUTIONS = {
    # Centuries 0-2 (0-299 years)
    (0, 2): {
        Region.ASIA_MINOR: 0.70,
        Region.LEVANT: 0.25,
        Region.EGYPT: 0.05,
    },
    # Centuries 3-5 (300-599 years)
    (3, 5): {
        Region.ASIA_MINOR: 0.55,
        Region.LEVANT: 0.25,
        Region.EGYPT: 0.20,
    },
    # Century 6 onwards (>= 600 years)
    (6, None): {  # None indicates "onwards"
        Region.ASIA_MINOR: 1.00,
        Region.LEVANT: 0.00,
        Region.EGYPT: 0.00,
    },
}


def _get_regional_distribution_for_century(century: int) -> Dict[Region, float]:
    """Determines the appropriate regional distribution for a given century."""
    for (start_century, end_century), distribution in REGIONAL_DISTRIBUTIONS.items():
        if end_century is None:  # Century 6 onwards
            if century >= start_century:
                return distribution
        elif start_century <= century <= end_century:
            return distribution
    # Default to the latest distribution if century is beyond defined ranges
    return REGIONAL_DISTRIBUTIONS[(6, None)]


def _allocate_demand(tick: int, aggregate_demand: int) -> Dict[Region, int]:
    """
    Deterministically allocates aggregate demand across regions based on the
    century of the current tick, using ceiling rounding for each region.

    Args:
        tick: The current simulation tick (1 tick = 1 year).
        aggregate_demand: The total demand for new manuscripts at this tick.

    Returns:
        A dictionary mapping Region enums to allocated integer demand counts.
    """
    if aggregate_demand == 0:
        return {region: 0 for region in Region}

    century = tick // 100  # 1 tick = 1 year, so century = floor(tick / 100)
    distribution = _get_regional_distribution_for_century(century)

    regional_demand: Dict[Region, int] = {}
    for region, proportion in distribution.items():
        # Round up (ceiling) for each region independently
        regional_demand[region] = ceil(aggregate_demand * proportion)

    return regional_demand


def handle_deaths(state: GenerationState) -> GenerationState:
    """Processes manuscript deaths for the current tick.

    This function identifies manuscripts whose scheduled `death_tick` has
    arrived. For each of these, it sets the `death_reason` to `NATURAL` if
    it has not already been set by another event (e.g., persecution). Finally,
    it removes the manuscripts from the `alive_manuscripts` set.

    Args:
        state (GenerationState): The current state of the genealogy generation.

    Returns:
        GenerationState: The updated state.
    """
    current_tick = state.tick
    manuscript_registry = state.registries.manuscripts

    dead_manuscripts = {ms_id for ms_id in state.alive_manuscripts if manuscript_registry.get(ms_id).death_tick == current_tick}

    if dead_manuscripts:
        for ms_id in dead_manuscripts:
            manuscript = manuscript_registry.get(ms_id)
            # Only set to NATURAL if a reason hasn't been set by a more
            # specific event (like persecution) in the same tick.
            if manuscript.death_reason is None:
                manuscript.death_reason = DeathReason.NATURAL
        state.alive_manuscripts -= dead_manuscripts

    return state


def handle_migration(
    state: GenerationState,
    rng: RNG,
    p_region_migration: float,
    p_internal_relocation: Optional[float] = None,
) -> GenerationState:
    """Handles the migration of manuscripts between and within regions.

    Migration is a manuscript-level event. It affects the manuscript's `region`
    and `location` attributes. This change can influence future exemplar
    selection for new manuscripts spawned in the affected regions, but it does
    not alter the historical genealogy graph.

    Args:
        state: The current simulation state.
        rng: The seeded random number generator.
        p_region_migration: Probability of a manuscript migrating to a
                            different region.
        p_internal_relocation: Probability of a manuscript relocating within
                               its current region.

    Returns:
        The updated simulation state.
    """
    # Iterate over a copy as the underlying registry objects will be modified
    for ms_id in list(state.alive_manuscripts):
        manuscript = state.registries.manuscripts.get(ms_id)

        # 1. Check for region migration
        if rng.random() < p_region_migration:
            other_regions = [r for r in Region if r != manuscript.region]
            if other_regions:
                new_region_value = rng.choice([r.value for r in other_regions])
                new_region = Region(new_region_value)  # Convert back to Enum
                manuscript.region = new_region
                manuscript.location = generate_random_coordinates(new_region, rng)
            continue  # A manuscript can only have one migration event per tick

        # 2. Check for internal relocation
        if p_internal_relocation and rng.random() < p_internal_relocation:
            manuscript.location = generate_random_coordinates(manuscript.region, rng)

    return state


def _spawn_new_manuscripts_from_demand(
    state: GenerationState,
    demand_today: Dict[Region, int],
    params: SimulationConfig,
    rng: RNG,
    material_transition_manager: MaterialTransitionManager,
    script_transition_manager: ScriptTransitionManager,
) -> GenerationState:
    """Spawns new manuscripts to meet exogenous demand.

    This function evaluates regional demand for manuscripts at the current tick
    and creates new manuscripts if the number of currently alive manuscripts
    in a region ("stock") is less than the demand.

    For each spawned manuscript, it also creates the associated witness and
    adds a new root node to the genealogy graph, representing the witness
    instance.

    The relationship between these entities is:
    `Manuscript -> Witness -> WitnessInstance (Graph Node)`
    - A Manuscript is the physical object with metadata (region, material).
    - A Witness is the textual content tied to a Manuscript.
    - A WitnessInstance is an abstract node in the genealogy graph,
      representing the manuscript's existence at a point in time.

    Args:
        state: The current simulation state.
        demand_today: A dictionary defining regional demand for the current tick.
        death_ticks: A queue of pre-calculated death ticks for new manuscripts.
        params: The validated simulation configuration object.
        rng: The random number generator.
        material_transition_manager: The manager for time-dependent material probabilities.
        script_transition_manager: The manager for time-dependent script probabilities.

    Returns:
        The updated simulation state.
    """
    current_tick = state.tick
    if not demand_today:
        return state

    # Count alive manuscripts per region
    stock = {region: 0 for region in Region}
    # Also collect them for the exemplar selection step
    alive_by_region: Dict[Region, List[Manuscript]] = {region: [] for region in Region}
    for ms_id in state.alive_manuscripts:
        manuscript = state.registries.manuscripts.get(ms_id)
        stock[manuscript.region] += 1
        alive_by_region[manuscript.region].append(manuscript)

    # Pre-build KDTrees for each region with alive manuscripts, once per tick
    kdtree_by_region: Dict[Region, KDTree] = {}
    for region, alive_manuscripts_list in alive_by_region.items():
        if alive_manuscripts_list:
            locations = np.array([ms.location for ms in alive_manuscripts_list])
            kdtree_by_region[region] = KDTree(locations)

    # Evaluate demand and spawn
    for region, demanded_count in demand_today.items():
        stock_count = stock.get(region, 0)
        if demanded_count > stock_count:
            # Get the pre-built KDTree for the current region, if available
            region_kdtree = kdtree_by_region.get(region)
            # Spawn new manuscripts
            for _ in range(demanded_count - stock_count):
                # 1. Create Manuscript
                manuscript_id = f"M{next(state.manuscript_id_counter)}"
                location = generate_random_coordinates(region, rng)
                reputation = sample_reputation(rng, params.reputation_distribution)

                material = material_transition_manager.get_material_for_tick(current_tick, rng)

                # Probabilistically determine lifespan and death tick
                lifespan = sample_lifespan(material=material, region=region, rng=rng)
                death_tick = current_tick + lifespan

                manuscript = Manuscript(
                    manuscript_id=manuscript_id,
                    birth_tick=current_tick,
                    death_tick=death_tick,
                    material=material,
                    region=region,
                    location=location,
                )
                state.registries.manuscripts.add(manuscript)

                # 2. Select Exemplars
                exemplars = select_exemplars(
                    new_manuscript=manuscript,
                    alive_manuscripts_in_region=alive_by_region[region],
                    graph=state.graph,
                    manuscript_to_instance_map=state.manuscript_to_instance_map,
                    rng=rng,
                    kdtree=region_kdtree,  # Pass the pre-built KDTree
                )

                # 3. Create Witness and WitnessInstance (Graph Node)
                witness_id = f"W{next(state.witness_id_counter)}"
                instance_id = f"I{next(state.witness_instance_id_counter)}"
                script = script_transition_manager.get_script_for_tick(current_tick, rng)
                witness = Witness(
                    witness_id=witness_id,
                    manuscript_id=manuscript_id,
                    script=script,
                )
                state.registries.witnesses.add(witness)

                if not exemplars:  # This covers cases where `select_exemplars` found nothing
                    if state.graph.number_of_nodes() == 0:
                        # This is the very first node, the autograph.
                        add_root_node(
                            graph=state.graph,
                            node_id=instance_id,
                            witness_id=witness_id,
                            manuscript_id=manuscript_id,
                            birth_tick=current_tick,
                            reputation=reputation,
                        )
                        new_text = make_initial_text(params)
                        state.registries.instance_texts[instance_id] = new_text
                    else:
                        # No local exemplars found, but the graph is not empty.
                        # Select parents from all alive instances.
                        all_alive_instance_ids = [
                            state.manuscript_to_instance_map[ms_id]
                            for ms_id in state.alive_manuscripts
                            if ms_id in state.manuscript_to_instance_map  # Safety check
                        ]

                        if not all_alive_instance_ids:
                            # This should ideally not happen if the graph is not empty,
                            # but if it does, it implies a bug or extreme scenario
                            raise ValueError("No alive instances to select parents from.")

                        # Determine number of parents (1-3), replicating logic from exemplar_selection
                        num_parents_choice = rng.choice([1, 2, 3], p=[0.8, 0.1, 0.1])
                        # Ensure we don't try to pick more parents than available
                        num_parents = min(num_parents_choice, len(all_alive_instance_ids))

                        # Select parents randomly from all alive instances
                        random_parents = rng.choice(
                            all_alive_instance_ids,
                            size=num_parents,
                            replace=False,  # Ensure unique parents
                        ).tolist()  # Convert numpy array to list for add_child_node

                        add_child_node(
                            graph=state.graph,
                            node_id=instance_id,
                            parent_node_ids=random_parents,
                            witness_id=witness_id,
                            manuscript_id=manuscript_id,
                            birth_tick=current_tick,
                            reputation=reputation,
                        )
                        exemplar_texts = [state.registries.instance_texts[eid] for eid in random_parents]
                        new_text = apply_scribal_rule(
                            exemplar_texts=exemplar_texts,
                            rng=rng,
                            reputation=reputation,
                            config=params,
                        )
                        state.registries.instance_texts[instance_id] = new_text
                else:  # Exemplars were found by select_exemplars
                    add_child_node(
                        graph=state.graph,
                        node_id=instance_id,
                        parent_node_ids=exemplars,
                        witness_id=witness_id,
                        manuscript_id=manuscript_id,
                        birth_tick=current_tick,
                        reputation=reputation,
                    )
                    # This is a copy, so generate its text from exemplars
                    exemplar_texts = [state.registries.instance_texts[eid] for eid in exemplars]
                    new_text = apply_scribal_rule(
                        exemplar_texts=exemplar_texts,
                        rng=rng,
                        reputation=reputation,
                        config=params,
                    )
                    state.registries.instance_texts[instance_id] = new_text

                # 4. Update state
                state.alive_manuscripts.add(manuscript_id)
                state.manuscript_to_instance_map[manuscript_id] = instance_id

    return state


def advance_tick(
    state: GenerationState,
    demand_today: Dict[Region, int],
    params: SimulationConfig,
    rng: RNG,
    event_manager: HistoricalEventManager,
    material_transition_manager: MaterialTransitionManager,
    script_transition_manager: ScriptTransitionManager,
    log_tick_frequency: int,
) -> GenerationState:
    """Advances the simulation clock and orchestrates per-tick events.

    Args:
        state: The current state of the genealogy generation.
        demand_today: A dictionary defining regional demand for the current tick.
        death_ticks: A queue of pre-calculated death ticks.
        params: The validated simulation configuration object.
        rng: The random number generator for this simulation.
        event_manager: The manager for historical events.
        material_transition_manager: The manager for time-dependent material probabilities.
        script_transition_manager: The manager for time-dependent script probabilities.

    Returns:
        The updated state after processing the tick.
    """
    state.tick += 1

    log_level = logging.INFO
    current_tick_for_logging = state.tick

    if current_tick_for_logging % log_tick_frequency == 0:
        logger.log(log_level, f"Tick {current_tick_for_logging:04d}: Starting... (Alive Manuscripts: {len(state.alive_manuscripts)})")

    start_time = time.perf_counter()

    # 1. Process deaths (mechanistic)
    state = handle_deaths(state)

    # 2. Process historical events (exogenous shocks)
    event_manager.apply_events_for_tick(state, rng)

    # 3. Process migration (mechanistic)
    state = handle_migration(
        state=state,
        rng=rng,
        p_region_migration=params.p_region_migration,
        p_internal_relocation=params.p_internal_relocation,
    )

    # 4. Spawn new manuscripts based on demand (mechanistic)
    state = _spawn_new_manuscripts_from_demand(
        state,
        demand_today,
        params,
        rng,
        material_transition_manager,
        script_transition_manager,
    )

    # 5. Record telemetry
    state.telemetry.append({
        "tick": state.tick,
        "alive_manuscripts": len(state.alive_manuscripts),
        "total_manuscripts": len(state.registries.manuscripts),
    })

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    if current_tick_for_logging % log_tick_frequency == 0:
        logger.log(
            log_level,
            f"Tick {current_tick_for_logging:04d}: Completed in {duration_ms:.2f}ms. "
            f"(Alive Manuscripts: {len(state.alive_manuscripts)}, "
            f"Total Manuscripts: {len(state.registries.manuscripts)})",
        )

    return state


def run_genealogy_generator(parameters: Dict[str, Any], rng: RNG) -> GenerationState:
    """High-level orchestration entry point for genealogy generation.

    This function drives the entire deterministic, tick-based process of
    generating a manuscript genealogy graph. It initialises the simulation
    state and then iterates through the specified number of ticks, calling
    `advance_tick` for each step.

    The generation process is deterministic: given the same `parameters` and a
    identically-seeded `rng`, it will always produce the exact same genealogy.

    Args:
        parameters (Dict[str, Any]): A dictionary of simulation parameters,
                                     validated against `SimulationConfig`.
        rng (RNG): A seeded NumPy random number generator to ensure
                   reproducibility.

    Returns:
        nx.DiGraph: The final generated genealogy graph, where nodes represent
                    witness instances and edges represent copying events.
    """
    # 1. Validate and structure parameters
    config = SimulationConfig(**parameters)

    # 2. Initialize state and managers
    state = initialise_generation_state()

    event_configs = [p.model_dump() for p in config.persecutions]
    for event_config in event_configs:
        event_config["event_type"] = "persecution"

    event_manager = HistoricalEventManager(event_configs)

    material_transition_manager = MaterialTransitionManager([m.model_dump() for m in config.material_transitions])
    script_transition_manager = ScriptTransitionManager([s.model_dump() for s in config.script_transitions])

    # 3. Run simulation loop
    for _ in range(config.total_ticks):
        current_tick = state.tick + 1  # Demand is calculated for the *next* tick
        # Get aggregate demand for the current tick, using last known value if not explicitly defined
        aggregate_demand_for_tick = config.demand_schedule.root.get(
            current_tick, config.demand_schedule.root.get(max(config.demand_schedule.root.keys()), 0)
        )

        # Allocate aggregate demand to regions
        demand_today = _allocate_demand(current_tick, aggregate_demand_for_tick)

        state = advance_tick(
            state=state,
            demand_today=demand_today,
            params=config,
            rng=rng,
            event_manager=event_manager,
            material_transition_manager=material_transition_manager,
            script_transition_manager=script_transition_manager,
            log_tick_frequency=config.log_tick_frequency,
        )

    return state
