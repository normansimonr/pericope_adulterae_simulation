import json
import shutil
from enum import Enum
from pathlib import Path

import numpy as np
import pydantic  # For Pydantic models


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
