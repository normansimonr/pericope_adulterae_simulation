import csv
from pathlib import Path
from typing import List


def serialize_majority_text(majority_segments: List[int]) -> str:
    """
    Converts a list of segment integers to a string, preserving leading zeros.
    If the list is empty, returns an empty string.
    """
    if not majority_segments:
        return ""
    return "".join(map(str, majority_segments))


def write_temp_result(
    experiment_root: Path,
    run_id: int,
    run_seed: int,
    regime: str,
    total_manuscripts_spawned: int,
    majority_text_segments: List[int],
):
    """
    Writes a single-row CSV file to a temporary directory for later aggregation.
    Parallel-safe because each run/regime has a unique filename.
    """
    temp_dir = experiment_root / "temp_results"
    temp_dir.mkdir(parents=True, exist_ok=True)

    filename = f"run_{run_id}_{regime}.csv"
    file_path = temp_dir / filename

    majority_text_str = serialize_majority_text(majority_text_segments)

    fieldnames = ["run_id", "run_seed", "regime", "total_manuscripts_spawned", "majority_text"]

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "run_seed": run_seed,
            "regime": regime,
            "total_manuscripts_spawned": total_manuscripts_spawned,
            "majority_text": majority_text_str,
        })
