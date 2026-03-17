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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from numpy.random import Generator as RNG
from scipy.spatial import KDTree  # type: ignore

from pasim.config.schema import SimulationConfig
from pasim.core.exemplar_selection import select_exemplars
from pasim.core.genealogy import add_child_node, add_root_node, validate_genealogy_full
from pasim.core.genealogy_snapshot import GenealogyNode, GenealogySnapshot
from pasim.core.historical_events import HistoricalEventManager
from pasim.core.material_transition_manager import MaterialTransitionManager
from pasim.core.reputation import sample_inherited_reputation
from pasim.core.script_transition_manager import ScriptTransitionManager
from pasim.core.simulation_state import GenerationState, initialise_generation_state
from pasim.core.spatial import generate_random_coordinates
from pasim.core.state import DeathReason, Manuscript, Region, Script, Witness

logger = logging.getLogger(__name__)


def _get_regional_distribution_for_century(century: int, config: SimulationConfig) -> Dict[Region, float]:
    """Determines the appropriate regional distribution for a given century."""
    distributions = config.regional_demand_distributions

    # Map century to the keys in distributions (0-2, 3-5, 6+)
    target_key = "6+"
    if 0 <= century <= 2:
        target_key = "0-2"
    elif 3 <= century <= 5:
        target_key = "3-5"

    dist_map = distributions.get(target_key, distributions.get("6+"))
    if dist_map is None:
        # Fallback if somehow 6+ is also missing
        return {Region.ASIA_MINOR: 1.0}

    # Map string keys to Region enum
    return {Region(k): v for k, v in dist_map.items()}


def _allocate_demand(tick: int, aggregate_demand: int, config: SimulationConfig) -> Dict[Region, int]:
    """
    Deterministically allocates aggregate demand across regions based on the
    century of the current tick, using ceiling rounding for each region.

    Args:
        tick: The current simulation tick (1 tick = 1 year).
        aggregate_demand: The total demand for new manuscripts at this tick.
        config: Simulation configuration.

    Returns:
        A dictionary mapping Region enums to allocated integer demand counts.
    """
    if aggregate_demand == 0:
        return {region: 0 for region in Region}

    century = tick // 100  # 1 tick = 1 year, so century = floor(tick / 100)
    distribution = _get_regional_distribution_for_century(century, config)

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
            # Update alive_by_region mapping
            state.alive_by_region[manuscript.region].remove(ms_id)
        state.alive_manuscripts -= dead_manuscripts

    return state


def _migrate_between_regions(
    state: GenerationState,
    alive_list: List[str],
    p_migration: float,
    rng: RNG,
    config: SimulationConfig,
) -> List[str]:
    """Handles migration of manuscripts between different regions."""
    n_migrate = rng.binomial(len(alive_list), p_migration)
    if n_migrate <= 0:
        return []

    migrating_ids = rng.choice(alive_list, size=n_migrate, replace=False).tolist()
    for ms_id in migrating_ids:
        manuscript = state.registries.manuscripts.get(ms_id)
        old_region = manuscript.region
        other_regions = [r for r in Region if r != old_region]
        if other_regions:
            new_region = rng.choice(cast(Any, other_regions))
            manuscript.region = new_region
            manuscript.location = generate_random_coordinates(new_region, rng, config)

            # Update alive_by_region mapping
            state.alive_by_region[old_region].remove(ms_id)
            state.alive_by_region[new_region].add(ms_id)
    return migrating_ids


def _relocate_internally(
    state: GenerationState,
    remaining_ids: List[str],
    p_relocation: float,
    rng: RNG,
    config: SimulationConfig,
) -> None:
    """Handles internal relocation of manuscripts within their current regions."""
    if p_relocation <= 0 or not remaining_ids:
        return

    n_relocate = rng.binomial(len(remaining_ids), p_relocation)
    if n_relocate > 0:
        relocating_ids = rng.choice(remaining_ids, size=n_relocate, replace=False).tolist()
        for ms_id in relocating_ids:
            manuscript = state.registries.manuscripts.get(ms_id)
            manuscript.location = generate_random_coordinates(manuscript.region, rng, config)


def handle_migration(
    state: GenerationState,
    rng: RNG,
    config: SimulationConfig,
) -> GenerationState:
    """Handles the migration of manuscripts between and within regions.

    Migration is a manuscript-level event. It affects the manuscript's `region`
    and `location` attributes. This change can influence future exemplar
    selection for new manuscripts spawned in the affected regions, but it does
    not alter the historical genealogy graph.

    Args:
        state: The current simulation state.
        rng: The seeded random number generator.
        config: The simulation configuration.

    Returns:
        The updated simulation state.
    """
    if not state.alive_manuscripts:
        return state

    alive_list = list(state.alive_manuscripts)

    # 1. Identify which manuscripts migrate between regions
    migrating_ids = _migrate_between_regions(state, alive_list, config.p_region_migration, rng, config)

    # 2. Identify which manuscripts relocate internally
    remaining_ids = list(state.alive_manuscripts - set(migrating_ids))
    _relocate_internally(state, remaining_ids, config.p_internal_relocation, rng, config)

    return state


def _get_regional_alive_data(state: GenerationState) -> tuple[Dict[Region, KDTree], Dict[Region, List[Manuscript]]]:
    """Pre-builds KDTrees and list of alive manuscripts for each region."""
    kdtree_by_region: Dict[Region, KDTree] = {}
    alive_by_region_objects: Dict[Region, List[Manuscript]] = {}

    for region, ms_ids in state.alive_by_region.items():
        if ms_ids:
            alive_manuscripts_list = [state.registries.manuscripts.get(ms_id) for ms_id in ms_ids]
            alive_by_region_objects[region] = alive_manuscripts_list
            locations = np.array([ms.location for ms in alive_manuscripts_list])
            kdtree_by_region[region] = KDTree(locations)
    return kdtree_by_region, alive_by_region_objects


def _generate_batch_spawn_properties(
    n_to_spawn: int,
    region: Region,
    current_tick: int,
    params: SimulationConfig,
    rng: RNG,
    material_transition_manager: MaterialTransitionManager,
    script_transition_manager: ScriptTransitionManager,
) -> Dict[str, Any]:
    """Generates a batch of properties for manuscripts to be spawned."""
    bounds = params.region_bounds.get(region.value)
    if bounds is None:
        raise ValueError(f"No bounds defined for region: {region.value}")
    x_bounds, y_bounds = bounds[0], bounds[1]

    locations_x = rng.uniform(x_bounds[0], x_bounds[1], size=n_to_spawn)
    locations_y = rng.uniform(y_bounds[0], y_bounds[1], size=n_to_spawn)

    material_dist = material_transition_manager.get_active_distribution(current_tick)
    batch_materials = rng.choice(material_dist["materials"], p=material_dist["probabilities"], size=n_to_spawn)

    script_dist = script_transition_manager.get_active_distribution(current_tick)
    batch_scripts = rng.choice(script_dist["scripts"], p=script_dist["probabilities"], size=n_to_spawn)

    # Vectorized lifespan generation
    batch_lifespans = np.zeros(n_to_spawn, dtype=int)
    lp = params.lifespan_parameters
    if lp:
        for mat in material_dist["materials"]:
            mask = batch_materials == mat
            if np.any(mask):
                from .lifespan import _get_lognormal_params

                mu, sigma = _get_lognormal_params(mat, region, params)
                lifespans = rng.lognormal(mean=mu, sigma=sigma, size=np.sum(mask))
                batch_lifespans[mask] = np.maximum(1, np.floor(lifespans)).astype(int)

    return {
        "locations_x": locations_x,
        "locations_y": locations_y,
        "materials": batch_materials,
        "scripts": batch_scripts,
        "lifespans": batch_lifespans,
    }


def _handle_spawned_witness_node(
    state: GenerationState,
    instance_id: str,
    witness_id: str,
    manuscript_id: str,
    current_tick: int,
    reputation: int,
    exemplars: List[str],
    params: SimulationConfig,
    rng: RNG,
) -> None:
    """Adds a new witness instance node to the genealogy graph."""
    if exemplars:
        add_child_node(
            graph=state.graph,
            node_id=instance_id,
            parent_node_ids=exemplars,
            witness_id=witness_id,
            manuscript_id=manuscript_id,
            birth_tick=current_tick,
            reputation=reputation,
        )
        return

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
    else:
        # This branch should rarely be hit now that parent selection is handled in _spawn_new_manuscripts_from_demand
        raise RuntimeError("No exemplars provided for non-root node.")


def _handle_cultural_replacement(
    state: GenerationState,
    child_script: Script,
    exemplars: List[str],
    current_tick: int,
    params: SimulationConfig,
    rng: RNG,
    instance_id: str,
) -> None:
    """Potentially kills Uncial parents when a Minuscule child is born."""
    if child_script != Script.MINUSCULE or params.p_uncial_exemplar_death_on_minuscule_birth <= 0:
        return

    for parent_id in exemplars:
        parent_ms_id = state.graph.nodes[parent_id]["manuscript_id"]
        if parent_ms_id in state.alive_manuscripts:
            parent_witness_id = state.graph.nodes[parent_id]["witness_id"]
            parent_witness = state.registries.witnesses.get(parent_witness_id)
            if parent_witness.script == Script.UNCIAL:
                if rng.random() < params.p_uncial_exemplar_death_on_minuscule_birth:
                    state.alive_manuscripts.remove(parent_ms_id)
                    parent_ms = state.registries.manuscripts.get(parent_ms_id)
                    state.alive_by_region[parent_ms.region].remove(parent_ms_id)
                    parent_ms.death_tick = current_tick
                    parent_ms.death_reason = DeathReason.CULTURAL_REPLACEMENT
                    logger.debug(f"Tick {current_tick}: Uncial {parent_ms_id} killed by Minuscule {instance_id} birth")


def _get_intervention_targets(config: SimulationConfig) -> List[tuple[int, Region]]:
    """Identifies all unique (year, region) pairs where a PA intervention is planned."""
    targets = set()
    # Check top-level defaults
    targets.add((config.pa_intervention_year, config.pa_intervention_region))

    # Check per-regime overrides
    if config.pa_regime_configs:
        for regime_cfg in config.pa_regime_configs.values():
            targets.add((regime_cfg.pa_intervention_year, regime_cfg.pa_intervention_region))

    return list(targets)


def _select_parents_and_reputation(
    state: GenerationState,
    manuscript: Manuscript,
    region_kdtree: Optional[KDTree],
    alive_by_region_objects: Dict[Region, List[Manuscript]],
    params: SimulationConfig,
    rng: RNG,
) -> Tuple[List[str], int]:
    """Selects parents and determines reputation for a new manuscript."""
    region = manuscript.region
    exemplars = select_exemplars(
        new_manuscript=manuscript,
        alive_manuscripts_in_region=alive_by_region_objects.get(region, []),
        graph=state.graph,
        manuscript_to_instance_map=state.manuscript_to_instance_map,
        rng=rng,
        config=params,
        kdtree=region_kdtree,
        reputation_cache=state.instance_reputations,
    )

    # Determine reputation via inheritance
    if exemplars:
        parent_reps = [int(state.instance_reputations[pid]) for pid in exemplars]
        reputation = sample_inherited_reputation(parent_reps, rng)
    elif state.graph.number_of_nodes() == 0:
        # Autograph: always max reputation
        reputation = 5
    else:
        # No local exemplars, will inherit from random global parents
        all_alive_instance_ids = [
            state.manuscript_to_instance_map[ms_id] for ms_id in state.alive_manuscripts if ms_id in state.manuscript_to_instance_map
        ]
        num_parents_choice = rng.choice([1, 2, 3], p=params.parent_num_distribution)
        num_parents = min(num_parents_choice, len(all_alive_instance_ids))
        exemplars = rng.choice(all_alive_instance_ids, size=num_parents, replace=False).tolist()
        parent_reps = [int(state.instance_reputations[pid]) for pid in exemplars]
        reputation = sample_inherited_reputation(parent_reps, rng)

    return exemplars, reputation


def _spawn_new_manuscripts_from_demand(
    state: GenerationState,
    demand_today: Dict[Region, int],
    params: SimulationConfig,
    rng: RNG,
    material_transition_manager: MaterialTransitionManager,
    script_transition_manager: ScriptTransitionManager,
) -> GenerationState:
    """Spawns new manuscripts to meet exogenous demand."""
    current_tick = state.tick
    if not demand_today:
        return state

    kdtree_by_region, alive_by_region_objects = _get_regional_alive_data(state)
    intervention_targets = _get_intervention_targets(params)

    spawned_this_tick: List[str] = []
    for region, demanded_count in demand_today.items():
        stock_count = len(state.alive_by_region[region])
        n_to_spawn = demanded_count - stock_count

        if (current_tick, region) in intervention_targets and demanded_count > 0:
            n_to_spawn = max(n_to_spawn, 1)

        if n_to_spawn <= 0:
            continue

        region_kdtree = kdtree_by_region.get(region)
        props = _generate_batch_spawn_properties(
            n_to_spawn, region, current_tick, params, rng, material_transition_manager, script_transition_manager
        )

        for i in range(n_to_spawn):
            manuscript_id = f"M{next(state.manuscript_id_counter)}"
            manuscript = Manuscript(
                manuscript_id=manuscript_id,
                birth_tick=current_tick,
                death_tick=current_tick + props["lifespans"][i],
                material=props["materials"][i],
                region=region,
                location=(props["locations_x"][i], props["locations_y"][i]),
            )
            state.registries.manuscripts.add(manuscript)

            exemplars, reputation = _select_parents_and_reputation(state, manuscript, region_kdtree, alive_by_region_objects, params, rng)

            witness_id = f"W{next(state.witness_id_counter)}"
            instance_id = f"I{next(state.witness_instance_id_counter)}"
            state.registries.witnesses.add(Witness(witness_id=witness_id, manuscript_id=manuscript_id, script=props["scripts"][i]))

            _handle_spawned_witness_node(state, instance_id, witness_id, manuscript_id, current_tick, reputation, exemplars, params, rng)
            _handle_cultural_replacement(state, props["scripts"][i], exemplars, current_tick, params, rng, instance_id)

            state.alive_manuscripts.add(manuscript_id)
            state.alive_by_region[region].add(manuscript_id)
            state.manuscript_to_instance_map[manuscript_id] = instance_id
            state.instance_reputations[instance_id] = float(reputation)
            spawned_this_tick.append(instance_id)

    _tag_innovator_nodes(state, spawned_this_tick, params)
    return state


def _tag_innovator_nodes(state: GenerationState, spawned_ids: List[str], config: SimulationConfig) -> None:
    """Identifies and tags the innovator node for each active PA regime."""
    if not spawned_ids:
        return

    targets = _get_intervention_targets_by_regime(config)

    # Filter to regimes that have a target today
    active_regimes = {regime: target for regime, target in targets.items() if state.tick == target[0]}
    if not active_regimes:
        return

    # Group spawned nodes by region for efficient lookup
    spawned_by_region = _group_spawned_by_region(state, spawned_ids)

    for regime, (_, target_region) in active_regimes.items():
        eligible_ids = spawned_by_region.get(target_region, [])
        if eligible_ids:
            _apply_innovator_tag(state, regime, eligible_ids, target_region, config)


def _group_spawned_by_region(state: GenerationState, spawned_ids: List[str]) -> Dict[Region, List[str]]:
    """Groups spawned instance IDs by their birth region."""
    spawned_by_region: Dict[Region, List[str]] = {}
    for inst_id in spawned_ids:
        ms_id = state.graph.nodes[inst_id]["manuscript_id"]
        region = state.registries.manuscripts.get(ms_id).region
        spawned_by_region.setdefault(region, []).append(inst_id)
    return spawned_by_region


def _apply_innovator_tag(
    state: GenerationState, regime: str, eligible_ids: List[str], target_region: Region, config: SimulationConfig
) -> None:
    """Selects and tags the innovator for a specific regime."""
    best_id = _select_innovator_for_regime(state, eligible_ids, target_region, config)
    if best_id:
        if "pa_intervention_regimes" not in state.graph.nodes[best_id]:
            state.graph.nodes[best_id]["pa_intervention_regimes"] = []
        state.graph.nodes[best_id]["pa_intervention_regimes"].append(regime)


def _get_intervention_targets_by_regime(config: SimulationConfig) -> Dict[str, Tuple[int, Region]]:
    """Generates the (year, region) target for each PA regime."""
    targets: Dict[str, Tuple[int, Region]] = {
        "insertion": (config.pa_intervention_year, config.pa_intervention_region),
        "omission": (config.pa_intervention_year, config.pa_intervention_region),
    }

    if config.pa_regime_configs:
        for regime in ["insertion", "omission"]:
            # Explicitly cast to the Literal required by the Pydantic field's keys
            r_key = cast(Literal["insertion", "omission"], regime)
            if r_key in config.pa_regime_configs:
                cfg = config.pa_regime_configs[r_key]
                targets[regime] = (cfg.pa_intervention_year, cfg.pa_intervention_region)

    return targets


def _select_innovator_for_regime(
    state: GenerationState, eligible_ids: List[str], target_region: Region, config: SimulationConfig
) -> Optional[str]:
    """Selects the best innovator from a pool of eligible nodes based on density."""
    # Find nodes born in the same region (reference pool for density)
    reference_ms_ids = list(state.alive_by_region[target_region])
    reference_locations = np.array([state.registries.manuscripts.get(ms_id).location for ms_id in reference_ms_ids])
    ref_tree = KDTree(reference_locations)

    best_id = None
    max_density = -1

    # We must ensure we sort eligible_ids to be deterministic for tie-breaking
    for inst_id in sorted(eligible_ids):
        ms_id = state.graph.nodes[inst_id]["manuscript_id"]
        loc = state.registries.manuscripts.get(ms_id).location
        # Density = number of neighbors within radius
        density = ref_tree.query_ball_point(loc, r=config.pa_intervention_radius, return_length=True)

        if density > max_density:
            max_density = density
            best_id = inst_id
        elif density == max_density:
            # Tie-break: lowest numeric ID
            if best_id is None or int(inst_id[1:]) < int(best_id[1:]):
                best_id = inst_id

    return best_id


def advance_tick(
    state: GenerationState,
    demand_today: Dict[Region, int],
    params: SimulationConfig,
    rng: RNG,
    event_manager: HistoricalEventManager,
    material_transition_manager: MaterialTransitionManager,
    script_transition_manager: ScriptTransitionManager,
    log_tick_frequency: int,
    run_id: int = 1,
    attempt: int = 0,
) -> GenerationState:
    """Advances the simulation clock and orchestrates per-tick events.

    Args:
        state: The current state of the genealogy generation.
        demand_today: A dictionary defining regional demand for the current tick.
        params: The validated simulation configuration object.
        rng: The random number generator for this simulation.
        event_manager: The manager for historical events.
        material_transition_manager: The manager for time-dependent material probabilities.
        script_transition_manager: The manager for time-dependent script probabilities.
        log_tick_frequency: The frequency (in ticks) at which progress updates.
        run_id: The identifier for the current simulation run.
        attempt: The retry attempt number for the current run.

    Returns:
        The updated state after processing the tick.
    """
    state.tick += 1

    log_level = logging.INFO
    current_tick_for_logging = state.tick

    if current_tick_for_logging % log_tick_frequency == 0:
        logger.log(
            log_level,
            f"Run {run_id} (att: {attempt}) | Tick {current_tick_for_logging:04d}: Starting... "
            f"(Alive Manuscripts: {len(state.alive_manuscripts)})",
        )

    start_time = time.perf_counter()

    # 1. Process deaths (mechanistic)
    state = handle_deaths(state)

    # 2. Process historical events (exogenous shocks)
    event_manager.apply_events_for_tick(state, rng)

    # 3. Process migration (mechanistic)
    state = handle_migration(
        state=state,
        rng=rng,
        config=params,
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
            f"Run {run_id} (att: {attempt}) | Tick {current_tick_for_logging:04d}: Completed in {duration_ms:.2f}ms. "
            f"(Alive Manuscripts: {len(state.alive_manuscripts)}, "
            f"Total Manuscripts: {len(state.registries.manuscripts)})",
        )

    # Perform full genealogy validation periodically, if enabled
    validation_frequency = params.validation_frequency
    if validation_frequency > 0 and state.tick % validation_frequency == 0:
        logger.debug(f"Tick {state.tick}: Performing periodic full genealogy validation...")
        validate_genealogy_full(state.graph)
        logger.debug(f"Tick {state.tick}: Periodic full genealogy validation passed.")

    return state


def extract_genealogy_snapshot(state: GenerationState) -> GenealogySnapshot:
    """
    Extracts a serialisable snapshot of the demographic genealogy from the simulation state.
    """
    nodes = []
    # Sort by birth tick to be clean
    sorted_node_ids = sorted(state.graph.nodes, key=lambda n: (state.graph.nodes[n]["birth_tick"], n))

    for node_id in sorted_node_ids:
        node_data = state.graph.nodes[node_id]
        manuscript = state.registries.manuscripts.get(node_data["manuscript_id"])
        witness = state.registries.witnesses.get(node_data["witness_id"])

        parent_ids = list(state.graph.predecessors(node_id))

        nodes.append(
            GenealogyNode(
                instance_id=node_id,
                witness_id=node_data["witness_id"],
                manuscript_id=node_data["manuscript_id"],
                birth_tick=node_data["birth_tick"],
                death_tick=manuscript.death_tick,
                parent_ids=parent_ids,
                region=manuscript.region,
                material=manuscript.material,
                script=witness.script,
                reputation=node_data["reputation"],
                location=manuscript.location,
                pa_intervention_regimes=node_data.get("pa_intervention_regimes", []),
            )
        )

    return GenealogySnapshot(nodes=nodes)


def run_genealogy_generator(
    config: Union[SimulationConfig, Dict[str, Any]],
    rng: RNG,
    run_id: int = 1,
    attempt: int = 0,
) -> GenerationState:
    """High-level orchestration entry point for genealogy generation.

    This function drives the entire deterministic, tick-based process of
    generating a manuscript genealogy graph. It initialises the simulation
    state and then iterates through the specified number of ticks, calling
    `advance_tick` for each step.

    The generation process is deterministic: given the same `config` and an
    identically-seeded `rng`, it will always produce the exact same genealogy.

    Args:
        config (Union[SimulationConfig, Dict[str, Any]]): The simulation configuration.
        rng (RNG): A seeded NumPy random number generator to ensure
                   reproducibility.
        run_id (int): The identifier for the current simulation run.
        attempt (int): The retry attempt number for the current run.

    Returns:
        GenerationState: The final generated simulation state, including the
                        genealogy graph and registries.
    """
    # 1. Initialize config if it's a dict
    if isinstance(config, dict):
        config = SimulationConfig(**config)

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
        if current_tick in config.demand_schedule.root:
            aggregate_demand_for_tick = config.demand_schedule.root[current_tick]
        else:
            # Find the largest tick in the schedule that is <= current_tick
            past_ticks = [t for t in config.demand_schedule.root.keys() if t <= current_tick]
            if past_ticks:
                aggregate_demand_for_tick = config.demand_schedule.root[max(past_ticks)]
            else:
                # If no past ticks, default to the value of the earliest defined tick
                # or 0 if the schedule is empty (which shouldn't happen due to validation)
                earliest_tick = min(config.demand_schedule.root.keys())
                aggregate_demand_for_tick = config.demand_schedule.root[earliest_tick]

        # Allocate aggregate demand to regions
        demand_today = _allocate_demand(current_tick, aggregate_demand_for_tick, config)

        state = advance_tick(
            state=state,
            demand_today=demand_today,
            params=config,
            rng=rng,
            event_manager=event_manager,
            material_transition_manager=material_transition_manager,
            script_transition_manager=script_transition_manager,
            log_tick_frequency=config.log_tick_frequency,
            run_id=run_id,
            attempt=attempt,
        )

    return state
