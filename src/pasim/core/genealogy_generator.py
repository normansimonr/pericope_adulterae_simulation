"""
Genealogy Generator with Death and Demand-Based Spawning.

This module provides a tick-based orchestration for generating a manuscript
genealogy. It defines the core control flow, state representation, and interfaces
for the generation process, including:
- Manuscript "death" handling.
- Demand-based manuscript spawning.

The generator's design enforces a strict separation of concerns:
- **Genealogy Graph**: Nodes are abstract "witness instances." The graph topology
  (who copied from whom) is its sole concern.
- **Manuscript Registry**: Holds the rich metadata for physical manuscripts,
  such as region, material, and birth/death ticks.

This separation ensures that attributes of the physical artifacts do not pollute
the abstract genealogical structure.

Explicit Exclusions:
- Exemplar selection policies.
- Migration, contamination, and scribal error models.
- Textual state (tagged strings).
- Batch execution or file I/O.
"""
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, MutableSet, Deque

import networkx as nx
from numpy.random import Generator as RNG

from pasim.core.genealogy import create_empty_genealogy, add_root_node
from pasim.core.state import StateRegistry, Manuscript, Witness, Region, Material


@dataclass
class GenerationState:
    """Encapsulates the complete state for a genealogy generation run."""
    tick: int
    graph: nx.DiGraph
    registries: StateRegistry
    alive_manuscripts: MutableSet[str]
    # Counters for generating unique IDs
    manuscript_id_counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    witness_id_counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    witness_instance_id_counter: itertools.count = field(default_factory=lambda: itertools.count(1))


def initialise_generation_state() -> GenerationState:
    """Creates and returns an empty genealogy generation state.

    This function initialises the state for a new simulation run, setting the
    tick to zero and preparing empty data structures.

    Returns:
        GenerationState: An object representing the pristine initial state.
    """
    return GenerationState(
        tick=0,
        graph=create_empty_genealogy(),
        registries=StateRegistry(),
        alive_manuscripts=set(),
    )


def handle_deaths(state: GenerationState) -> GenerationState:
    """Processes manuscript deaths for the current tick.

    This function identifies manuscripts whose scheduled `death_tick` has
    arrived and removes them from the set of `alive_manuscripts`. This action
    is purely administrative; it affects which manuscripts are available for
    future copying events but does not alter the historical record.

    The genealogy graph and manuscript registry are not modified. This
    deterministically separates the concept of "alive" (available for copying)
    from "exists" (part of the historical record).

    Args:
        state (GenerationState): The current state of the genealogy generation.

    Returns:
        GenerationState: The updated state.
    """
    current_tick = state.tick
    manuscript_registry = state.registries.manuscripts
    
    dead_manuscripts = {
        ms_id
        for ms_id in state.alive_manuscripts
        if manuscript_registry.get(ms_id).death_tick == current_tick
    }

    if dead_manuscripts:
        state.alive_manuscripts -= dead_manuscripts

    return state


def _spawn_new_manuscripts_from_demand(
    state: GenerationState,
    demand: Dict[int, Dict[Region, int]],
    death_ticks: Deque[int],
    rng: RNG,
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
        demand: A dictionary mapping tick -> region -> demanded count.
        death_ticks: A queue of pre-calculated death ticks for new manuscripts.
        rng: The random number generator.

    Returns:
        The updated simulation state.
    """
    current_tick = state.tick
    demand_today = demand.get(current_tick, {})
    if not demand_today:
        return state

    # Count alive manuscripts per region
    stock = {region: 0 for region in Region}
    for ms_id in state.alive_manuscripts:
        manuscript = state.registries.manuscripts.get(ms_id)
        stock[manuscript.region] += 1
    
    # Evaluate demand and spawn
    for region, demanded_count in demand_today.items():
        stock_count = stock.get(region, 0)
        if demanded_count > stock_count:
            # Spawn new manuscripts
            for _ in range(demanded_count - stock_count):
                if not death_ticks:
                    raise ValueError("Ran out of pre-generated death ticks.")
                
                # 1. Create Manuscript
                manuscript_id = f"M{next(state.manuscript_id_counter)}"
                manuscript = Manuscript(
                    manuscript_id=manuscript_id,
                    birth_tick=current_tick,
                    death_tick=death_ticks.popleft(),
                    material=rng.choice(list(Material)),
                    region=region,
                    location=(rng.uniform(0, 1), rng.uniform(0, 1)),
                )
                state.registries.manuscripts.add(manuscript)
                state.alive_manuscripts.add(manuscript_id)

                # 2. Create Witness
                witness_id = f"W{next(state.witness_id_counter)}"
                witness = Witness(
                    witness_id=witness_id, manuscript_id=manuscript_id
                )
                state.registries.witnesses.add(witness)

                # 3. Create WitnessInstance (Graph Node)
                instance_id = f"I{next(state.witness_instance_id_counter)}"
                add_root_node(
                    graph=state.graph,
                    node_id=instance_id,
                    witness_id=witness_id,
                    manuscript_id=manuscript_id,
                    birth_tick=current_tick,
                    death_tick=None, # Per design, node does not store death tick
                )

    return state


def advance_tick(
    state: GenerationState,
    demand: Dict[int, Dict[Region, int]],
    death_ticks: Deque[int],
    rng: RNG
) -> GenerationState:
    """Advances the simulation clock and orchestrates per-tick events.

    Args:
        state: The current state of the genealogy generation.
        demand: A dictionary defining regional demand per tick.
        death_ticks: A queue of pre-calculated death ticks.
        rng: The random number generator for this simulation.

    Returns:
        The updated state after processing the tick.
    """
    state.tick += 1

    # 1. Process deaths
    state = handle_deaths(state)

    # 2. Spawn new manuscripts based on demand
    state = _spawn_new_manuscripts_from_demand(state, demand, death_ticks, rng)
    
    # --- Placeholder for future logic ---
    # 3. Process migration.
    # 4. Select exemplars and create copies (child nodes).

    return state


def run_genealogy_generator(parameters: Dict[str, Any], rng: RNG) -> nx.DiGraph:
    """High-level orchestration entry point for genealogy generation.

    This function drives the entire deterministic, tick-based process of
    generating a manuscript genealogy graph. It initialises the simulation
    state and then iterates through the specified number of ticks, calling
    `advance_tick` for each step.

    The generation process is deterministic: given the same `parameters` and a
    identically-seeded `rng`, it will always produce the exact same genealogy.

    Args:
        parameters (Dict[str, Any]): A dictionary of simulation parameters.
            Must include:
            - 'total_ticks': Total number of ticks to simulate.
            - 'demand': Dictionary mapping tick -> region -> count.
            - 'death_ticks': An iterable of pre-calculated death ticks.
        rng (RNG): A seeded NumPy random number generator to ensure
                   reproducibility.

    Returns:
        nx.DiGraph: The final generated genealogy graph, where nodes represent
                    witness instances and edges represent copying events.
    """
    state = initialise_generation_state()
    total_ticks = parameters.get("total_ticks", 0)
    demand = parameters.get("demand", {})
    # Use a deque for efficient popleft()
    death_ticks = deque(parameters.get("death_ticks", []))

    for _ in range(total_ticks):
        state = advance_tick(state, demand, death_ticks, rng)

    return state.graph
