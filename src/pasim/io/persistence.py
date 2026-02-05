import json
import shutil
from enum import Enum
from pathlib import Path

import numpy as np
import pydantic  # For Pydantic models

from pasim.execution.runner import SimulationResult  # For accessing simulation results


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
        # Handle Pydantic models (v2 prefers model_dump, v1 uses dict())
        if isinstance(obj, pydantic.BaseModel):
            if hasattr(obj, "model_dump"):  # Pydantic v2
                return obj.model_dump()
            else:  # Pydantic v1
                return obj.dict()
        return super().default(obj)


def _resolve_run_directory(params_path: Path) -> Path:
    """
    Determines the next run directory path and ensures its existence.

    Given a params_path like 'experiments/exp001_baseline/params.yaml',
    it will create a directory like 'experiments/exp001_baseline/runs/<run_id>/'.

    Args:
        params_path: Path to the experiment's parameters file.

    Returns:
        The Path to the newly created (or re-created) run directory.
    """
    experiment_dir = params_path.parent
    runs_dir = experiment_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    existing_run_numbers = []
    for item in runs_dir.iterdir():
        if item.is_dir():
            try:
                # Attempt to convert folder name to an integer
                existing_run_numbers.append(int(item.name))
            except ValueError:
                # Ignore folders that are not integers
                continue

    next_run_number = 1
    if existing_run_numbers:
        next_run_number = max(existing_run_numbers) + 1

    run_dir = runs_dir / str(next_run_number)

    # If the directory already exists, delete it completely and recreate it
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    return run_dir


def _save_config(run_dir: Path, params_path: Path):
    """
    Saves a copy of the input config file to the run directory.
    """
    shutil.copy(params_path, run_dir / "config.yaml")


def _save_run_metadata(run_dir: Path, result: SimulationResult):
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


def _save_genealogy(run_dir: Path, result: SimulationResult):
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


def _save_instances(run_dir: Path, result: SimulationResult):
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


def _save_manuscripts(run_dir: Path, result: SimulationResult):
    """
    Saves the full manuscript registry to a JSON file.
    """
    manuscripts_data = []
    for _, manuscript in result.state.registries.manuscripts.items():
        manuscripts_data.append(manuscript)  # CustomJsonEncoder handles Pydantic models
    with open(run_dir / "manuscripts.json", "w") as f:
        json.dump(manuscripts_data, f, indent=2, cls=CustomJsonEncoder)


def save_run(result: SimulationResult, params_path: str):
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
    _save_genealogy(run_dir, result)  # New call
    _save_instances(run_dir, result)  # New call
    _save_manuscripts(run_dir, result)  # New call
