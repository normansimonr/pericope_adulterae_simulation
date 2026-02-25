import dataclasses  # For handling dataclasses
import json
import shutil
import time  # Import time for a small backoff
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, cast  # Added TYPE_CHECKING

import numpy as np
import pydantic  # For Pydantic models

from pasim.core.state import DeathReason

if TYPE_CHECKING:  # Added conditional import
    from pasim.execution.runner import SimulationResult


class CustomJsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing various types not natively supported by JSON,
    including Path objects, Enums, NumPy scalars/arrays, and Pydantic models.
    """

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        # Handle Pydantic models (v2 prefers model_dump, v1 uses dict())
        if isinstance(obj, pydantic.BaseModel):
            if hasattr(obj, "model_dump"):  # Pydantic v2
                return obj.model_dump()
            else:  # Pydantic v1
                return obj.model_dump()
        return super().default(obj)


# ... (rest of the file remains the same until _resolve_run_directory) ...


def _resolve_run_directory(params_path: Path) -> Path:
    """
    Determines a unique run directory path and ensures its existence.
    Robustly handles concurrent calls from parallel processes.

    Given a params_path like 'experiments/exp000_baseline/params.yaml',
    it will create a directory like 'experiments/exp000_baseline/runs/<run_id>/'.

    Args:
        params_path: Path to the experiment's parameters file.

    Returns:
        The Path to the newly created unique run directory.
    """
    experiment_dir = params_path.parent
    runs_dir = experiment_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)  # Ensure parent 'runs' directory exists

    # Implement a retry loop to find a unique run number in a concurrent-safe way
    max_retries = 10
    for _ in range(max_retries):
        existing_run_numbers = []
        for item in runs_dir.iterdir():
            if item.is_dir():
                try:
                    existing_run_numbers.append(int(item.name))
                except ValueError:
                    continue  # Ignore non-integer named directories

        next_run_number = 1
        if existing_run_numbers:
            next_run_number = max(existing_run_numbers) + 1

        run_dir = runs_dir / str(next_run_number)

        try:
            run_dir.mkdir(parents=True)  # Attempt to create the directory exclusively
            return run_dir  # Success! Return the unique directory
        except FileExistsError:
            # Directory was created by another process concurrently.
            # Loop again to find the next available number.
            time.sleep(0.01)  # Small backoff to reduce contention

    raise RuntimeError(f"Failed to create a unique run directory after {max_retries} retries in {runs_dir}")


def _save_config(run_dir: Path, params_path: Path):
    """
    Saves a copy of the input config file to the run directory.
    """
    shutil.copy(params_path, run_dir / "config.yaml")


def _save_run_metadata(run_dir: Path, result: "SimulationResult"):
    """
    Saves high-level simulation metadata to a JSON file.
    """
    graph = result.graph
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Total manuscripts: all ever created, regardless of death
    # Accessing .manuscripts directly from the StateRegistry
    total_manuscripts = len(result.state.registries.manuscripts)
    # Total instances: all nodes in the genealogy graph
    total_instances = num_nodes

    metadata = {
        "seed": result.seed,
        "final_tick": result.state.tick,
        "total_instances": total_instances,
        "total_manuscripts": total_manuscripts,
        "graph_nodes": num_nodes,
        "graph_edges": num_edges,
    }
    with open(run_dir / "run_metadata.json", "w") as f:
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


def _save_instance_texts(run_dir: Path, result: "SimulationResult"):
    """
    Saves all instance texts in TSV format.
    """
    file_path = run_dir / "instance_texts.tsv"
    with open(file_path, "w") as f:
        if not result.state.registries.instance_texts:
            return  # No texts to save

        # Determine text length and generate header
        first_instance_id = next(iter(result.state.registries.instance_texts))
        text_length = len(result.state.registries.instance_texts[first_instance_id])
        header = "instance_id\t" + "\t".join(f"token_{i}" for i in range(text_length))
        f.write(header + "\n")

        # Get instance IDs and their birth ticks, then sort
        instance_birth_ticks = []
        for node_id, data in result.graph.nodes(data=True):
            instance_birth_ticks.append((node_id, data["birth_tick"]))
        instance_birth_ticks.sort(key=lambda x: x[1])  # Sort by birth_tick

        # Write texts in order
        for instance_id, _ in instance_birth_ticks:
            text_array = result.state.registries.instance_texts.get(instance_id)
            if text_array is not None:
                text_str = "\t".join(map(str, text_array.tolist()))
                f.write(f"{instance_id}\t{text_str}\n")


def _save_telemetry(run_dir: Path, result: "SimulationResult"):
    """
    Saves telemetry data to a JSON file.
    """
    with open(run_dir / "telemetry.json", "w") as f:
        json.dump(result.state.telemetry, f, indent=2, cls=CustomJsonEncoder)


def _save_events_log(run_dir: Path, result: "SimulationResult"):
    """
    Generates and saves a chronological log of key simulation events.
    Infers events from the final state of the simulation result.
    """
    events = []

    # Collect manuscript birth and death events from the manuscript registry
    for ms_id, manuscript in result.state.registries.manuscripts.items():
        events.append({"tick": manuscript.birth_tick, "type": "manuscript_birth", "id": ms_id})

        if manuscript.death_tick is not None and manuscript.death_tick != float("inf"):
            # Use the new `death_reason` field to determine event type
            if manuscript.death_reason == DeathReason.PERSECUTION:
                event_type = "manuscript_destroyed"
            else:  # Covers NATURAL and None as a fallback
                event_type = "manuscript_death"
            events.append({"tick": manuscript.death_tick, "type": event_type, "id": ms_id})

    # Infer instance birth events (from graph nodes)
    for node_id, data in result.graph.nodes(data=True):
        parents = list(result.graph.predecessors(node_id))
        event_info = {"tick": data["birth_tick"], "type": "instance_birth", "id": node_id, "parents": parents}
        events.append(event_info)

    # Sort events chronologically
    events.sort(key=lambda x: (x["tick"], x["type"]))

    with open(run_dir / "events.log", "w") as f:
        for event in events:
            if event["type"] == "manuscript_birth":
                f.write(f"[TICK {event['tick']}] Manuscript {event['id']} created\n")
            elif event["type"] == "manuscript_death":
                f.write(f"[TICK {event['tick']}] Manuscript {event['id']} died\n")
            elif event["type"] == "manuscript_destroyed":
                f.write(f"[TICK {event['tick']}] Manuscript {event['id']} destroyed\n")
            elif event["type"] == "instance_birth":
                if event["parents"]:
                    parent_info = ", ".join(map(str, cast(List[str], event["parents"])))
                    f.write(f"[TICK {event['tick']}] Instance {event['id']} created from {parent_info}\n")
                else:
                    f.write(f"[TICK {event['tick']}] Instance {event['id']} created (autograph)\n")


def save_run(result: "SimulationResult", params_path: str):
    """
    Public entry point to save the essential simulation output for reproducibility.

    Args:
        result: The SimulationResult object containing all simulation outputs.
        params_path: The path to the original parameters file.
    """
    params_path_obj = Path(params_path)
    run_dir = _resolve_run_directory(params_path_obj)

    _save_config(run_dir, params_path_obj)
    _save_run_metadata(run_dir, result)
    _save_genealogy(run_dir, result)
    _save_instances(run_dir, result)
    _save_manuscripts(run_dir, result)
    _save_instance_texts(run_dir, result)
    _save_telemetry(run_dir, result)
    _save_events_log(run_dir, result)  # New call
