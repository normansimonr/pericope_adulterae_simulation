import dataclasses  # For handling dataclasses
import json
import logging
import shutil
import time  # Import time for a small backoff
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, cast  # Added TYPE_CHECKING

import numpy as np
import pydantic  # For Pydantic models

from pasim.analysis.majority_text import compute_majority_text, save_majority_text
from pasim.core.state import DeathReason
from pasim.core.survivor_sampler import save_sampling_results

if TYPE_CHECKING:  # Added conditional import
    from pasim.execution.runner import SimulationResult


logger = logging.getLogger(__name__)


class CustomJsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing various types not natively supported by JSON,
    including Path objects, Enums, NumPy scalars/arrays, and Pydantic models.
    """

    def _handle_numpy(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return None

    def _handle_pydantic(self, obj):
        if isinstance(obj, pydantic.BaseModel):
            if hasattr(obj, "model_dump"):  # Pydantic v2
                return obj.model_dump()
            else:  # Pydantic v1
                return obj.model_dump()
        return None

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value

        numpy_res = self._handle_numpy(obj)
        if numpy_res is not None:
            return numpy_res

        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        pydantic_res = self._handle_pydantic(obj)
        if pydantic_res is not None:
            return pydantic_res

        return super().default(obj)


# ... (rest of the file remains the same until _resolve_run_directory) ...


def _get_existing_run_numbers(runs_dir: Path) -> list[int]:
    """Helper to list all existing run numbers in the runs directory."""
    existing_run_numbers = []
    if runs_dir.is_dir():
        for item in runs_dir.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                try:
                    existing_run_numbers.append(int(item.name[4:]))
                except ValueError:
                    continue  # Ignore malformed directories
    return existing_run_numbers


def resolve_run_directory(params_path: Path, create_dir: bool = True) -> Path:
    """
    Determines a unique run directory path. If create_dir is True, ensures its existence.
    Robustly handles concurrent calls from parallel processes.
    """
    experiment_dir = params_path.parent
    runs_dir = experiment_dir / "runs"

    if create_dir:
        runs_dir.mkdir(parents=True, exist_ok=True)

    max_retries = 10
    for _ in range(max_retries):
        existing_run_numbers = _get_existing_run_numbers(runs_dir)
        next_run_number = max(existing_run_numbers) + 1 if existing_run_numbers else 0
        run_dir = runs_dir / f"run_{next_run_number}"

        if not create_dir:
            return run_dir

        try:
            run_dir.mkdir(parents=True)
            return run_dir
        except FileExistsError:
            time.sleep(0.01)

    raise RuntimeError(f"Failed to create a unique run directory after {max_retries} retries in {runs_dir}")


def _save_config(run_dir: Path, params_path: Path):
    """
    Saves a copy of the input config file to the run directory.
    """
    shutil.copy(params_path, run_dir / "config.yaml")


def _save_demographic_metadata(run_dir: Path, result: "SimulationResult"):
    """
    Saves high-level demographic metadata to a JSON file in the run root.
    """
    graph = result.graph
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    total_manuscripts = len(result.state.registries.manuscripts)
    total_instances = num_nodes

    metadata = {
        "seed": result.seed,
        "final_tick": result.state.tick,
        "total_instances": total_instances,
        "total_manuscripts": total_manuscripts,
        "graph_nodes": num_nodes,
        "graph_edges": num_edges,
    }
    with open(run_dir / "demographic_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, cls=CustomJsonEncoder)


def _save_regime_metadata(regime_dir: Path, result: "SimulationResult", regime_name: str):
    """
    Saves regime-specific metadata including PA intervention details.
    """
    replay = result.replays[regime_name]

    metadata = {
        "run_id": result.run_id,
        "seed": result.seed,
        "pa_regime": replay.pa_regime,
        "pa_intervention_year": result.config.pa_intervention_year,
        "pa_intervention_region": result.config.pa_intervention_region,
        "pa_innovator_reputation": result.config.pa_innovator_reputation,
        "replay_seed": replay.seed,
    }
    with open(regime_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, cls=CustomJsonEncoder)


def _save_genealogy(run_dir: Path, result: "SimulationResult"):
    """
    Saves genealogy nodes and edges to a JSON file.
    """
    nodes_data = []
    for node_id, data in result.graph.nodes(data=True):
        nodes_data.append({
            "instance_id": node_id,
            "manuscript_id": data["manuscript_id"],
            "birth_tick": data["birth_tick"],
            "reputation": data["reputation"],
        })

    edges_data = []
    for u, v in result.graph.edges():
        edges_data.append({"parent": u, "child": v})

    genealogy = {"nodes": nodes_data, "edges": edges_data}
    with open(run_dir / "genealogy.json", "w") as f:
        json.dump(genealogy, f, indent=2, cls=CustomJsonEncoder)


def _save_instances(run_dir: Path, result: "SimulationResult"):
    """
    Saves all witness instance metadata to a JSON file.
    """
    instances_data = []
    for node_id, data in result.graph.nodes(data=True):
        instances_data.append({
            "instance_id": node_id,
            "manuscript_id": data["manuscript_id"],
            "witness_id": data["witness_id"],
            "birth_tick": data["birth_tick"],
            "reputation": data["reputation"],
        })
    with open(run_dir / "instances.json", "w") as f:
        json.dump(instances_data, f, indent=2, cls=CustomJsonEncoder)


def _save_manuscripts(run_dir: Path, result: "SimulationResult"):
    """
    Saves the full manuscript registry to a JSON file.
    """
    manuscripts_data = []
    for _, manuscript in result.state.registries.manuscripts.items():
        manuscripts_data.append(manuscript)  # CustomJsonEncoder handles Pydantic models
    with open(run_dir / "manuscripts.json", "w") as f:
        json.dump(manuscripts_data, f, indent=2, cls=CustomJsonEncoder)


def _get_first_available_text_length(result: "SimulationResult", texts: dict[str, np.ndarray]) -> int | None:
    """Finds the length of the first available survivor text."""
    for sid in result.survivor_sampling_result.sampled_witness_ids:
        if sid in texts:
            return len(texts[sid])
    return None


def _get_sorted_survivor_instances(result: "SimulationResult", survivor_ids: set[str]) -> list[tuple[str, int]]:
    """Returns a list of survivor instance IDs sorted by birth tick."""
    instance_birth_ticks = []
    for node_id, data in result.graph.nodes(data=True):
        if node_id in survivor_ids:
            instance_birth_ticks.append((node_id, data["birth_tick"]))
    instance_birth_ticks.sort(key=lambda x: x[1])
    return instance_birth_ticks


def _write_texts_to_file(f, instance_birth_ticks: list[tuple[str, int]], texts: dict[str, np.ndarray], text_length: int):
    """Writes instance texts to the provided file handle in TSV format."""
    header = "instance_id\t" + "\t".join(f"token_{i}" for i in range(text_length))
    f.write(header + "\n")

    for instance_id, _ in instance_birth_ticks:
        text_array = texts.get(instance_id)
        if text_array is not None:
            text_str = "\t".join(map(str, text_array.tolist()))
            f.write(f"{instance_id}\t{text_str}\n")


def _save_instance_texts(regime_dir: Path, result: "SimulationResult", texts: dict[str, np.ndarray]):
    """
    Saves sampled instance texts in TSV format for a specific regime.
    Also creates a dummy witnesses.parquet to satisfy path requirements.
    """
    file_path = regime_dir / "instance_texts.tsv"
    survivor_ids = set(result.survivor_sampling_result.sampled_witness_ids)

    with open(file_path, "w") as f:
        if not texts or not survivor_ids:
            pass
        else:
            text_length = _get_first_available_text_length(result, texts)
            if text_length is not None:
                instance_birth_ticks = _get_sorted_survivor_instances(result, survivor_ids)
                _write_texts_to_file(f, instance_birth_ticks, texts, text_length)

    (regime_dir / "witnesses.parquet").touch()


def _save_telemetry(run_dir: Path, result: "SimulationResult"):
    """
    Saves telemetry data to a JSON file.
    """
    with open(run_dir / "telemetry.json", "w") as f:
        json.dump(result.state.telemetry, f, indent=2, cls=CustomJsonEncoder)


def _collect_manuscript_events(result: "SimulationResult") -> list[dict]:
    """Infers manuscript birth and death events from the registry."""
    events = []
    for ms_id, manuscript in result.state.registries.manuscripts.items():
        events.append({"tick": manuscript.birth_tick, "type": "manuscript_birth", "id": ms_id})
        if manuscript.death_tick is not None and manuscript.death_tick != float("inf"):
            event_type = "manuscript_destroyed" if manuscript.death_reason == DeathReason.PERSECUTION else "manuscript_death"
            events.append({"tick": manuscript.death_tick, "type": event_type, "id": ms_id})
    return events


def _collect_instance_events(result: "SimulationResult") -> list[dict]:
    """Infers instance birth events from the graph nodes."""
    events = []
    for node_id, data in result.graph.nodes(data=True):
        parents = list(result.graph.predecessors(node_id))
        events.append({"tick": data["birth_tick"], "type": "instance_birth", "id": node_id, "parents": parents})
    return events


def _write_event_to_log(f, event: dict):
    """Writes a single event entry to the log file handle."""
    tick, event_id = event["tick"], event["id"]
    if event["type"] == "manuscript_birth":
        f.write(f"[TICK {tick}] Manuscript {event_id} created\n")
    elif event["type"] == "manuscript_death":
        f.write(f"[TICK {tick}] Manuscript {event_id} died\n")
    elif event["type"] == "manuscript_destroyed":
        f.write(f"[TICK {tick}] Manuscript {event_id} destroyed\n")
    elif event["type"] == "instance_birth":
        if event["parents"]:
            parents = ", ".join(map(str, cast(List[str], event["parents"])))
            f.write(f"[TICK {tick}] Instance {event_id} created from {parents}\n")
        else:
            f.write(f"[TICK {tick}] Instance {event_id} created (autograph)\n")


def _save_events_log(run_dir: Path, result: "SimulationResult"):
    """
    Generates and saves a chronological log of key simulation events.
    Infers events from the final state of the simulation result.
    """
    events = _collect_manuscript_events(result)
    events.extend(_collect_instance_events(result))
    events.sort(key=lambda x: (x["tick"], x["type"]))

    with open(run_dir / "events.log", "w") as f:
        for event in events:
            _write_event_to_log(f, event)


def _save_genealogy_snapshot(run_dir: Path, result: "SimulationResult"):
    """
    Saves the full genealogy snapshot to a JSON file.
    """
    with open(run_dir / "genealogy_snapshot.json", "w") as f:
        json.dump(result.genealogy_snapshot.to_dict(), f, indent=2, cls=CustomJsonEncoder)


def save_demographics(result: "SimulationResult", run_dir: Path, params_path: Path):
    """
    Saves shared demographic and configuration data at the run root.
    """
    _save_config(run_dir, params_path)
    _save_demographic_metadata(run_dir, result)
    _save_genealogy(run_dir, result)
    _save_genealogy_snapshot(run_dir, result)
    _save_instances(run_dir, result)
    _save_manuscripts(run_dir, result)
    _save_telemetry(run_dir, result)
    _save_events_log(run_dir, result)

    target_total = result.config.survivor_sampling_targets.get("target_total", 3000)
    save_sampling_results(result.survivor_sampling_result, run_dir, target_total=target_total)


def save_replay(result: "SimulationResult", run_dir: Path, regime_name: str):
    """
    Saves regime-specific textual data in a subdirectory.
    """
    regime_dir = run_dir / regime_name
    regime_dir.mkdir(parents=True, exist_ok=True)

    replay = result.replays[regime_name]
    _save_regime_metadata(regime_dir, result, regime_name)
    _save_instance_texts(regime_dir, result, replay.instance_texts)

    # 3. Compute and save majority text
    survivor_ids = result.survivor_sampling_result.sampled_witness_ids
    if not survivor_ids:
        logger.warning(f"No survivors sampled for regime {regime_name}. Skipping majority text computation.")
        save_majority_text([], regime_dir)
    else:
        survivor_genomes = [replay.instance_texts[sid] for sid in survivor_ids if sid in replay.instance_texts]
        majority_segments = compute_majority_text(survivor_genomes)
        save_majority_text(majority_segments, regime_dir)


def save_run(result: "SimulationResult", params_path: str):
    """
    Public entry point to save the essential simulation output for reproducibility.
    Orchestrates saving shared demographic data and regime-specific textual data.

    Args:
        result: The SimulationResult object containing all simulation outputs.
        params_path: The path to the original parameters file.
    """
    params_path_obj = Path(params_path)
    run_dir = resolve_run_directory(params_path_obj)

    # 1. Save shared demographic and configuration data at the run root
    save_demographics(result, run_dir, params_path_obj)

    # 2. Save regime-specific data in subdirectories
    for regime_name in result.replays:
        save_replay(result, run_dir, regime_name)
