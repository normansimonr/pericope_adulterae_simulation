import csv
import shutil
import time
from pathlib import Path


def aggregate_results(experiment_root: Path):
    """
    Reads all temporary CSV files in experiment_root/temp_results/,
    merges them with any existing results.csv, sorts them, and deletes the temp dir.
    Uses a simple lock-file mechanism to prevent race conditions in parallel mode.
    """
    temp_dir = experiment_root / "temp_results"
    output_path = experiment_root / "results.csv"
    lock_path = experiment_root / "results.csv.lock"

    if not temp_dir.is_dir() or not any(temp_dir.glob("run_*.csv")):
        return

    # simple spin-lock
    max_retries = 100
    for _ in range(max_retries):
        try:
            with open(lock_path, "x"):
                # Lock acquired
                try:
                    _perform_aggregation(temp_dir, output_path)
                finally:
                    if lock_path.exists():
                        lock_path.unlink()
                return
        except FileExistsError:
            # Lock held by another process
            time.sleep(0.1)

    # If we got here, we couldn't get the lock
    raise RuntimeError(f"Could not acquire lock for results.csv at {lock_path} after {max_retries} retries.")


def _perform_aggregation(temp_dir: Path, output_path: Path):
    """Internal logic for result aggregation, called under lock."""
    all_rows = []
    fieldnames = [
        "run_id",
        "run_seed",
        "regime",
        "total_manuscripts_spawned",
        "majority_text",
        "pct_sampled_witnesses_with_pa",
        "pct_majority_disagree_autograph",
        "pct_all_witnesses_with_pa",
        "ideal_majority_text",
        "pct_ideal_majority_disagree_autograph",
    ]

    # 1. Load existing results if they exist
    if output_path.exists():
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _coerce_row_types(row)
                all_rows.append(row)

    # 2. Read temporary results
    merged_files = _load_temp_results(temp_dir, all_rows)

    if not all_rows:
        return

    # Sort by run_id, then regime
    all_rows.sort(key=lambda x: (x["run_id"], x["regime"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(all_rows)

    # 3. Cleanup processed temporary files
    for file_path in merged_files:
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass

    # Cleanup temp dir if it's empty
    if temp_dir.is_dir() and not any(temp_dir.iterdir()):
        shutil.rmtree(temp_dir)


def _load_temp_results(temp_dir: Path, all_rows: list) -> list[Path]:
    """Loads temporary CSV files and appends new rows to all_rows. Returns list of merged files."""
    merged_files = []
    # Use a set of (run_id, regime) tuples for O(N) duplicate checking
    existing_keys = {(row["run_id"], row["regime"]) for row in all_rows}

    for file_path in temp_dir.glob("run_*.csv"):
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                _coerce_row_types(row)
                key = (row["run_id"], row["regime"])
                if key not in existing_keys:
                    all_rows.append(row)
                    existing_keys.add(key)
                row_count += 1

            if row_count > 0:
                merged_files.append(file_path)
    return merged_files


def _coerce_row_types(row: dict):
    """In-place coercion of CSV string values to proper types."""
    row["run_id"] = int(row["run_id"])
    row["run_seed"] = int(row["run_seed"])
    row["total_manuscripts_spawned"] = int(row["total_manuscripts_spawned"])

    # Handle optional float fields
    for field in [
        "pct_sampled_witnesses_with_pa",
        "pct_majority_disagree_autograph",
        "pct_all_witnesses_with_pa",
        "pct_ideal_majority_disagree_autograph",
    ]:
        if field in row and row[field] != "":
            row[field] = float(row[field])
