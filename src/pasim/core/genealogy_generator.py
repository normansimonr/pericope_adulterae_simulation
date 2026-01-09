"""Genealogy Generator with Death Handling.

This module provides a tick-based orchestration structure for generating a
manuscript genealogy. It defines the core control flow, state representation,
and interfaces for the generation process, including the handling of
manuscript "deaths".

This implementation includes:
- A simulation loop and state management.
- Deterministic death handling: manuscripts are removed from the "alive" set
  at their scheduled death tick.

Explicit Exclusions in this implementation:
- Manuscript spawning and migration logic.
- Exemplar selection policies.
- Reputation assignment.
- Textual state (tagged strings).
- Batch execution or file I/O.
"""
from typing import Dict, Any, MutableSet

import networkx as nx
from numpy.random import Generator as RNG

from pasim.core.genealogy import create_empty_genealogy

# A lightweight dictionary-based representation for the genealogy state.
# - tick: The current simulation time-step.
# - graph: The directed acyclic graph (DAG) of manuscript relationships.
# - alive_manuscripts: A set of identifiers for manuscripts currently active.
GenealogyState = Dict[str, Any]


def initialise_genealogy_state() -> GenealogyState:
    """Creates and returns an empty genealogy generation state.

    This function initialises the state for a new simulation run, setting the
    tick to zero and preparing empty data structures for the genealogy graph
    and the set of alive manuscripts.

    Returns:
        GenealogyState: A dictionary representing the pristine initial state.
    """
    return {
        "tick": 0,
        "graph": create_empty_genealogy(),
        "alive_manuscripts": set(),
    }


def handle_deaths(state: GenealogyState) -> GenealogyState:
    """Processes manuscript deaths for the current tick.

    This function identifies manuscripts whose scheduled `death_tick` has
    arrived and removes them from the set of `alive_manuscripts`. This action
    is purely administrative; it affects which manuscripts are available for
    future copying events but does not alter the historical record.

    The genealogy graph itself is not modified. Nodes are never deleted,
    ensuring that the full, immutable history of the genealogy is preserved
    for post-hoc analysis. This deterministically separates the concept of
    "alive" (available for copying) from "exists" (part of the historical
    record).

    Args:
        state (GenealogyState): The current state of the genealogy generation.

    Returns:
        GenealogyState: The updated state with newly deceased manuscripts
                        removed from the `alive_manuscripts` set.
    """
    current_tick = state["tick"]
    graph = state["graph"]
    dead_manuscripts = {
        ms_id
        for ms_id in state["alive_manuscripts"]
        if graph.nodes[ms_id].get("death_tick") == current_tick
    }

    if dead_manuscripts:
        state["alive_manuscripts"] -= dead_manuscripts

    return state


def advance_tick(state: GenealogyState, rng: RNG) -> GenealogyState:
    """Advances the simulation clock and orchestrates per-tick events.

    This function serves as the main entry point for all events that occur
    within a single time-step. It first increments the tick, then calls
    sub-functions to handle discrete simulation events like deaths.

    Args:
        state (GenealogyState): The current state of the genealogy generation.
        rng (RNG): The random number generator for this simulation.

    Returns:
        GenealogyState: The updated state after processing the tick.
    """
    state["tick"] += 1

    # 1. Process deaths
    state = handle_deaths(state)

    # --- Placeholder for future logic ---
    # 2. Process migration (manuscripts moving between locations).
    # 3. Evaluate demand for new copies at each location.
    # 4. Spawn new manuscripts based on demand and exemplar availability.
    #    - This will involve exemplar selection, reputation assignment, and
    #      the creation of new witness nodes in the graph.

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
                                     Must include 'total_ticks'.
        rng (RNG): A seeded NumPy random number generator to ensure
                   reproducibility.

    Returns:
        nx.DiGraph: The final generated genealogy graph, where nodes represent
                    manuscript instances and edges represent copying events.
    """
    state = initialise_genealogy_state()
    total_ticks = parameters.get("total_ticks", 0)

    for _ in range(total_ticks):
        state = advance_tick(state, rng)

    return state["graph"]
