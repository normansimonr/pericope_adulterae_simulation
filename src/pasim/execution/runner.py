"""
This module provides the `run_single` function, which serves as the primary
entry point for executing a single, in-memory simulation run from a user-provided
parameter file. It orchestrates the setup, execution, and result aggregation
for a vertical slice of the `pasim` framework.
"""

import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import yaml

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_generator import extract_genealogy_snapshot, run_genealogy_generator
from pasim.core.genealogy_snapshot import GenealogySnapshot
from pasim.core.rng import RNGContext
from pasim.core.simulation_state import GenerationState
from pasim.core.survivor_sampler import SamplingResult, sample_survivors
from pasim.core.text_replay import TextReplayEngine
from pasim.io.persistence import resolve_run_directory, save_demographics, save_replay


@dataclass
class ReplayResult:
    """
    A container for the results of a single textual replay regime.
    """

    pa_regime: str
    instance_texts: Dict[str, np.ndarray]
    seed: int


@dataclass
class SimulationResult:
    """
    A structured container for the results of a single simulation run.
    This object provides a consistent interface for accessing the final state,
    genealogy graph, configuration, and other metadata from the simulation.
    Now supports multiple replay regimes for a single demographic history.
    """

    state: GenerationState
    graph: nx.DiGraph
    config: SimulationConfig
    seed: int
    genealogy_snapshot: GenealogySnapshot
    survivor_sampling_result: SamplingResult = field(default_factory=lambda: SamplingResult([], 0, {}))
    replays: Dict[str, ReplayResult] = field(default_factory=dict)


def derive_replay_seed(base_seed: int, regime: str) -> int:
    """
    Derives a deterministic replay seed from a base seed and regime name.
    Ensures that insertion/omission runs remain reproducible and different.
    """
    # Use adler32 for a simple, deterministic integer hash of the regime name
    regime_hash = zlib.adler32(regime.encode())
    # Combine with base_seed. Using bitwise XOR or simple addition is fine.
    # We ensure it's a 32-bit unsigned integer for compatibility.
    return (base_seed + regime_hash) & 0xFFFFFFFF


def run_single(params_path: str, seed: int = 20240105) -> SimulationResult:
    """
    Executes a single, in-memory simulation run.
    It splits the run into demographic simulation and dual text replay
    (insertion and omission regimes) over the same genealogy snapshot.
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

    # 4. Create RNG for demographic generation
    # This ensures demographic randomness is isolated from replay randomness.
    rng = RNGContext(seed).spawn(1)[0]

    # 5. Run demographic simulation
    # Builds the fixed genealogy graph (demographic scaffold)
    state = run_genealogy_generator(parameters=params_dict, rng=rng)

    # 6. Extract genealogy snapshot for replay
    snapshot = extract_genealogy_snapshot(state)

    # 7. Perform survivorship sampling (New Stage)
    sampling_result = sample_survivors(snapshot, seed, config.total_ticks)

    # 8. Initialize result container
    result = SimulationResult(
        state=state,
        graph=state.graph,
        config=config,
        seed=seed,
        genealogy_snapshot=snapshot,
        survivor_sampling_result=sampling_result,
    )

    # 9. Resolve run directory and save demographics BEFORE replay
    # This ensures that even if replay fails, the demographic scaffold is preserved.
    params_path_obj = Path(params_path)
    run_dir = resolve_run_directory(params_path_obj)
    save_demographics(result, run_dir, params_path_obj)

    # 10. Run text replay for both regimes and save them independently
    for regime in ["insertion", "omission"]:
        # Create a regime-specific config override
        regime_config = config.model_copy(update={"pa_regime": regime})

        # Derive a deterministic seed for this replay
        replay_seed = derive_replay_seed(seed, regime)

        # Run the replay engine
        replay_engine = TextReplayEngine(regime_config, snapshot, replay_seed)
        replayed_texts = replay_engine.run()

        # Store the result in the result object
        result.replays[regime] = ReplayResult(
            pa_regime=regime,
            instance_texts=replayed_texts,
            seed=replay_seed,
        )

        # Save this regime's specific output
        save_replay(result, run_dir, regime)

    return result
