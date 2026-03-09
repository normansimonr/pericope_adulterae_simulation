"""
This module provides the `run_single` function, which serves as the primary
entry point for executing a single, in-memory simulation run from a user-provided
parameter file. It orchestrates the setup, execution, and result aggregation
for both textual replay regimes.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

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
from pasim.io.results_writer import write_temp_result


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
    run_id: int = 1
    survivor_sampling_result: SamplingResult = field(default_factory=lambda: SamplingResult([], 0, {}))
    replays: Dict[str, ReplayResult] = field(default_factory=dict)


def derive_replay_seed(base_seed: int, regime: str) -> int:
    """
    Derives a deterministic seed for a textual replay regime from the base seed.
    This ensures that while each regime is deterministic, they use different
    random sequences for their respective mutations.
    """
    # Simple hash of the regime name to use as an offset
    import hashlib

    regime_hash = int(hashlib.sha256(regime.encode()).hexdigest(), 16) % (2**32)

    # Combine with base_seed. Using bitwise XOR or simple addition is fine.
    # We ensure it's a 32-bit unsigned integer for compatibility.
    return (base_seed + regime_hash) & 0xFFFFFFFF


def run_single(
    params_path: str,
    seed: int = 20240105,
    persistence_level: str = "full",
    run_id: Optional[int] = None,
    regime: Optional[str] = None,
) -> SimulationResult:
    """
    Executes a single, in-memory simulation run.
    It splits the run into demographic simulation and dual text replay
    (insertion and omission regimes) over the same genealogy snapshot.
    If a specific regime is provided, only that regime is replayed.
    """
    # 1. Validate path
    if "experiments/" not in params_path:
        raise ValueError("Parameter file must be located in the 'experiments/' directory.")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameter file not found at: {params_path}")

    # 2. Load parameters
    with open(params_path, "r") as f:
        params_dict = yaml.safe_load(f)
    if not isinstance(params_dict, dict):
        raise ValueError("YAML file must contain a dictionary.")

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

    # 7. Perform survivorship sampling
    sampling_result = sample_survivors(snapshot, seed, config.total_ticks)

    # 8. Resolve run directory and ID
    params_path_obj = Path(params_path)
    if run_id is None:
        # Fallback to dynamic resolution if run_id not provided
        run_dir = resolve_run_directory(params_path_obj, create_dir=(persistence_level == "full"))
        # Parse the ID from the 'run_N' folder name
        try:
            run_id = int(run_dir.name[4:])
        except (ValueError, IndexError):
            # Fallback for old directory names or errors
            run_id = 1
    else:
        # Use deterministic run_id-based folder if full persistence
        if persistence_level == "full":
            run_dir = params_path_obj.parent / "runs" / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = None

    # 9. Initialize result container
    result = SimulationResult(
        state=state,
        graph=state.graph,
        config=config,
        seed=seed,
        run_id=run_id,
        genealogy_snapshot=snapshot,
        survivor_sampling_result=sampling_result,
    )

    # 10. Save demographics if requested (only once per run_id)
    if persistence_level == "full" and run_dir:
        # To avoid race conditions in parallel mode, only the designated regime ('insertion')
        # writes the shared demographics. If no specific regime is provided, we also write.
        if regime is None or regime == "insertion":
            save_demographics(result, run_dir, params_path_obj)

    # 11. Run text replay for requested regimes
    target_regimes = [regime] if regime else ["insertion", "omission"]

    for r in target_regimes:
        # Create a regime-specific config override
        regime_config = config.model_copy(update={"pa_regime": r})

        # Derive a deterministic seed for this replay
        replay_seed = derive_replay_seed(seed, r)

        # Run the replay engine
        replay_engine = TextReplayEngine(regime_config, snapshot, replay_seed)
        replayed_texts = replay_engine.run()

        # Store the result in the result object
        result.replays[r] = ReplayResult(
            pa_regime=r,
            instance_texts=replayed_texts,
            seed=replay_seed,
        )

        # Save this regime's specific output if requested
        if persistence_level == "full" and run_dir:
            save_replay(result, run_dir, r)

        # 12. Always compute and write temporary results for aggregation
        from pasim.analysis.majority_text import compute_majority_text

        survivor_ids = result.survivor_sampling_result.sampled_witness_ids
        survivor_genomes = [replayed_texts[sid] for sid in survivor_ids if sid in replayed_texts]
        majority_segments = compute_majority_text(survivor_genomes)

        write_temp_result(
            experiment_root=params_path_obj.parent,
            run_id=run_id,
            run_seed=seed,
            regime=r,
            total_manuscripts_spawned=len(snapshot.nodes),
            majority_text_segments=majority_segments,
        )

    return result
