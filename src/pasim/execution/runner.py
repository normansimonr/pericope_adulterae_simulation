"""
This module provides the `run_single` function, which serves as the primary
entry point for executing a single, in-memory simulation run from a user-provided
parameter file. It orchestrates the setup, execution, and result aggregation
for a vertical slice of the `pasim` framework.
"""

import os
from dataclasses import dataclass

import networkx as nx
import yaml

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_generator import extract_genealogy_snapshot, run_genealogy_generator
from pasim.core.genealogy_snapshot import GenealogySnapshot
from pasim.core.rng import RNGContext
from pasim.core.simulation_state import GenerationState
from pasim.core.text_replay import TextReplayEngine
from pasim.io.persistence import save_run  # Import the new persistence function


@dataclass
class SimulationResult:
    """
    A structured container for the results of a single simulation run.
    This object provides a consistent interface for accessing the final state,
    genealogy graph, configuration, and other metadata from the simulation.
    """

    state: GenerationState
    graph: nx.DiGraph
    config: SimulationConfig
    seed: int
    genealogy_snapshot: GenealogySnapshot


def run_single(params_path: str, seed: int = 20240105) -> SimulationResult:
    """
    Executes a single, in-memory simulation run.
    It now splits the run into demographic simulation and dual text replay
    (insertion and omission regimes).
    """
    # 1. Validate path
    if "experiments/" not in params_path:
        raise ValueError("Parameter file must be located in the 'experiments/' directory.")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameter file not found at: {params_path}")

    # 2. Load parameters
    with open(params_path, "r") as f:
        params_dict = yaml.safe_load(f)

    # 3. Validate configuration
    config = SimulationConfig(**params_dict)

    # 4. Create RNG
    rng = RNGContext(seed).spawn(1)[0]

    # 5. Run demographic simulation
    # First, run the demographic layer to build the graph
    state = run_genealogy_generator(parameters=params_dict, rng=rng)

    # 6. Extract genealogy snapshot
    snapshot = extract_genealogy_snapshot(state)

    # 7. Run text replay layer for the configured regime
    # We use a separate seed for the replay to ensure independence from demographic randomness.
    replay_seed = seed + 1000
    replay_engine = TextReplayEngine(config, snapshot, replay_seed)
    replayed_texts = replay_engine.run()

    # Update state's instance_texts with replayed texts
    state.registries.instance_texts = replayed_texts

    # 8. Return structured result
    result = SimulationResult(state=state, graph=state.graph, config=config, seed=seed, genealogy_snapshot=snapshot)

    # Save the run results using the persistence layer
    save_run(result, params_path)

    return result
