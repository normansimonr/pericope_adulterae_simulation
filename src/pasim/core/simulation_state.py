"""
This module defines the GenerationState dataclass, which encapsulates the
complete state of a genealogy generation run. It has been extracted from
genealogy_generator.py to break circular import dependencies and centralize
the definition of the core simulation state.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict

import networkx as nx

from pasim.core.genealogy import create_empty_genealogy
from pasim.core.state import Region, StateRegistry


@dataclass
class GenerationState:
    """Encapsulates the complete state for a genealogy generation run.

    This dataclass holds all dynamic information for the simulation, including
    the genealogy graph, registries for all entities, and counters. The actual
    textual content of each witness instance is stored separately within the
    `registries` attribute, specifically in `state.registries.instance_texts`.
    """

    tick: int
    graph: nx.DiGraph
    registries: StateRegistry
    alive_manuscripts: set[str]
    # Maps each region to the set of manuscript IDs currently alive in that region
    alive_by_region: Dict[Region, set[str]] = field(default_factory=lambda: {r: set() for r in Region})
    manuscript_to_instance_map: Dict[str, Any] = field(default_factory=dict)
    # Cache for instance reputations to avoid graph lookups during exemplar selection
    instance_reputations: Dict[str, float] = field(default_factory=dict)
    # Cache for instance birth ticks for age-based sorting in exemplar selection
    instance_birth_ticks: Dict[str, int] = field(default_factory=dict)
    # Counters for generating unique IDs
    manuscript_id_counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    witness_id_counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    witness_instance_id_counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    telemetry: list = field(default_factory=list)


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
