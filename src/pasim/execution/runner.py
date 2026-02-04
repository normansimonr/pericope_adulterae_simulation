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
from pasim.core.genealogy_generator import run_genealogy_generator
from pasim.core.rng import RNGContext
from pasim.core.simulation_state import GenerationState


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


def run_single(params_path: str, seed: int = 20240105) -> SimulationResult:
    """
    Executes a single, in-memory simulation run.

    This function performs the following steps:
    1.  Validates that the provided `params_path` points to a real file
        within the `experiments/` directory.
    2.  Loads the YAML parameters from the file.
    3.  Validates the loaded parameters against the `SimulationConfig` schema.
    4.  Initializes a deterministic random number generator (`RNG`) from the
        provided `seed`.
    5.  Calls the core `run_genealogy_generator` to execute the simulation.
    6.  Bundles the final state, graph, config, and seed into a
        `SimulationResult` object.

    Args:
        params_path (str): The file path to the YAML configuration file.
                           Must be located within the `experiments/` directory.
        seed (int): The master seed for the random number generator to ensure
                    reproducibility. Defaults to a fixed value.

    Returns:
        SimulationResult: A dataclass containing the complete results of the run.

    Raises:
        ValueError: If `params_path` is invalid, does not exist, or is not
                    located within the `experiments/` directory.
        FileNotFoundError: If the specified `params_path` does not exist.
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

    # 5. Run simulation
    state = run_genealogy_generator(parameters=params_dict, rng=rng)

    # 6. Return structured result
    return SimulationResult(
        state=state,
        graph=state.graph,
        config=config,
        seed=seed,
    )
