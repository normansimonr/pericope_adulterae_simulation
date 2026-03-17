"""
This module provides the `run_single` function, which serves as the primary
entry point for executing a single, in-memory simulation run from a user-provided
parameter file. It orchestrates the setup, execution, and result aggregation
for both textual replay regimes.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

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
    innovator_id: str
    majority_text_segments: List[int]
    pct_sampled_witnesses_with_pa: Optional[float] = None
    pct_majority_disagree_autograph: Optional[float] = None
    pct_all_witnesses_with_pa: Optional[float] = None
    ideal_majority_text_segments: Optional[List[int]] = None
    pct_ideal_majority_disagree_autograph: Optional[float] = None


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


def _validate_and_load_config(params_path: str) -> SimulationConfig:
    """Validates the parameter file path and loads the configuration."""
    if "experiments/" not in params_path:
        raise ValueError("Parameter file must be located in the 'experiments/' directory.")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameter file not found at: {params_path}")

    with open(params_path, "r") as f:
        params_dict = yaml.safe_load(f)
    if not isinstance(params_dict, dict):
        raise ValueError("YAML file must contain a dictionary.")

    return SimulationConfig(**params_dict)


def _run_demographics(
    config: SimulationConfig, seed: int, run_id: int = 1, attempt: int = 0
) -> tuple[GenerationState, GenealogySnapshot, SamplingResult]:
    """Runs the demographic simulation and extracts the genealogy snapshot and survivor samples."""
    rng = RNGContext(seed).spawn(1)[0]
    state = run_genealogy_generator(config=config, rng=rng, run_id=run_id, attempt=attempt)
    snapshot = extract_genealogy_snapshot(state)
    sampling_result = sample_survivors(snapshot, seed, config.total_ticks, config)
    return state, snapshot, sampling_result


def _resolve_run_context(params_path_obj: Path, run_id: Optional[int], persistence_level: str) -> tuple[int, Optional[Path]]:
    """Resolves the run ID and directory based on persistence level."""
    if run_id is None:
        run_dir = resolve_run_directory(params_path_obj, create_dir=(persistence_level == "full"))
        try:
            resolved_run_id = int(run_dir.name[4:])
        except (ValueError, IndexError):
            resolved_run_id = 1
    else:
        resolved_run_id = run_id
        if persistence_level == "full":
            run_dir = params_path_obj.parent / "runs" / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = None
    return resolved_run_id, run_dir


@dataclass
class MinimalSimulationResult:
    """
    A memory-efficient container for simulation results.
    Used in 'minimal' persistence mode to avoid holding large graph/state objects.
    """

    run_id: int
    seed: int
    survivor_sampling_result: SamplingResult
    replays: Dict[str, ReplayResult] = field(default_factory=dict)


def _run_regime_replay(
    regime: str,
    config: SimulationConfig,
    seed: int,
    snapshot: GenealogySnapshot,
    result: Union[SimulationResult, MinimalSimulationResult],
    persistence_level: str,
    run_dir: Optional[Path],
    params_path_obj: Path,
) -> None:
    """Runs the text replay for a specific regime and stores/saves results."""
    from pasim.analysis.majority_text import compute_majority_text

    # Extract regime-specific parameters if they exist
    target_regime = cast(Literal["insertion", "omission"], regime)

    update_params: Dict[str, Any] = {"pa_regime": target_regime}
    if config.pa_regime_configs and target_regime in config.pa_regime_configs:
        regime_params = config.pa_regime_configs[target_regime]
        update_params.update({
            "pa_intervention_year": regime_params.pa_intervention_year,
            "pa_intervention_region": regime_params.pa_intervention_region,
            "pa_innovator_reputation": regime_params.pa_innovator_reputation,
        })

    regime_config = config.model_copy(update=update_params)
    replay_seed = derive_replay_seed(seed, regime)
    replay_engine = TextReplayEngine(regime_config, snapshot, replay_seed)

    # Use memory-efficient replay if we don't need full persistence
    survivor_ids = set(result.survivor_sampling_result.sampled_witness_ids)
    return_all = persistence_level == "full"
    replayed_texts = replay_engine.run(survivor_ids=survivor_ids, return_all_texts=return_all)

    # Metrics from on-the-fly tracking
    ideal_metrics = replay_engine.get_ideal_metrics()
    autograph_text = ideal_metrics["autograph_text"]
    text_length = config.text_length

    # Majority text of survivors
    survivor_genomes = [replayed_texts[sid] for sid in survivor_ids if sid in replayed_texts]
    majority_segments = compute_majority_text(survivor_genomes)

    # Metrics
    # 1. Percent of Sampled Witnesses Containing the PA
    pct_sampled_with_pa = None
    if survivor_genomes:
        count_sampled_with_pa = sum(1 for g in survivor_genomes if np.any(g != 0))
        pct_sampled_with_pa = count_sampled_with_pa / len(survivor_genomes)

    # 2. Percent of Majority Text Segments Disagreeing with the Autograph
    pct_majority_disagree = None
    if majority_segments and autograph_text is not None:
        majority_array = np.array(majority_segments, dtype=np.int16)
        disagree_count = np.sum(majority_array != autograph_text)
        pct_majority_disagree = disagree_count / text_length

    # 3. Percent of All Witnesses That Contain the PA (from incremental tracker)
    pct_all_with_pa = ideal_metrics["pct_all_witnesses_with_pa"]

    # 4. Ideal Majority Text (from incremental tracker)
    ideal_majority_segments = ideal_metrics["ideal_majority_text_segments"]

    # 5. Percent of Ideal Majority Segments Disagreeing with Autograph (from incremental tracker)
    pct_ideal_majority_disagree = ideal_metrics["pct_ideal_majority_disagree_autograph"]

    # The replay logic guarantees an innovator is found or it raises RuntimeError.
    assert replay_engine.innovator_id is not None

    result.replays[regime] = ReplayResult(
        pa_regime=regime,
        instance_texts=replayed_texts,  # This is filtered if persistence_level != "full"
        seed=replay_seed,
        innovator_id=replay_engine.innovator_id,
        majority_text_segments=majority_segments,
        pct_sampled_witnesses_with_pa=pct_sampled_with_pa,
        pct_majority_disagree_autograph=pct_majority_disagree,
        pct_all_witnesses_with_pa=pct_all_with_pa,
        ideal_majority_text_segments=ideal_majority_segments,
        pct_ideal_majority_disagree_autograph=pct_ideal_majority_disagree,
    )

    if persistence_level == "full" and run_dir:
        # mypy: 'result' is SimulationResult because persistence_level == "full"
        assert isinstance(result, SimulationResult)
        save_replay(result, run_dir, regime)

    write_temp_result(
        experiment_root=params_path_obj.parent,
        run_id=result.run_id,
        run_seed=seed,
        regime=regime,
        total_manuscripts_spawned=len(snapshot.nodes),
        majority_text_segments=majority_segments,
        pct_sampled_witnesses_with_pa=pct_sampled_with_pa,
        pct_majority_disagree_autograph=pct_majority_disagree,
        pct_all_witnesses_with_pa=pct_all_with_pa,
        ideal_majority_text_segments=ideal_majority_segments,
        pct_ideal_majority_disagree_autograph=pct_ideal_majority_disagree,
    )


def run_single(
    params_path: str,
    seed: int = 20240105,
    persistence_level: str = "full",
    run_id: Optional[int] = None,
    regime: Optional[str] = None,
    attempt: int = 0,
) -> Optional[Union[SimulationResult, MinimalSimulationResult]]:
    """
    Executes a single, in-memory simulation run.
    """
    config = _validate_and_load_config(params_path)
    params_path_obj = Path(params_path)
    resolved_run_id, run_dir = _resolve_run_context(params_path_obj, run_id, persistence_level)

    # 1. Run Demographics
    state, snapshot, sampling_result = _run_demographics(config, seed, run_id=resolved_run_id, attempt=attempt)

    result: Union[SimulationResult, MinimalSimulationResult]
    if persistence_level == "minimal":
        # Create lightweight result and DISCARD the heavy state/graph immediately
        result = MinimalSimulationResult(run_id=resolved_run_id, seed=seed, survivor_sampling_result=sampling_result)

        # Explicitly clear heavy objects
        del state
        import gc

        gc.collect()
    else:
        result = SimulationResult(
            state=state,
            graph=state.graph,
            config=config,
            seed=seed,
            run_id=resolved_run_id,
            genealogy_snapshot=snapshot,
            survivor_sampling_result=sampling_result,
        )

    if persistence_level == "full" and run_dir:
        if regime is None or regime == "insertion":
            # We know 'result' is a full SimulationResult here
            save_demographics(cast(SimulationResult, result), run_dir, params_path_obj)

    # 2. Run Text Replay
    target_regimes = [regime] if regime else ["insertion", "omission"]
    for r in target_regimes:
        _run_regime_replay(r, config, seed, snapshot, result, persistence_level, run_dir, params_path_obj)

    if persistence_level == "minimal":
        return None  # Still return None to prevent pickling back to main process

    return result
