import csv
import shutil
from pathlib import Path


def aggregate_results(experiment_root: Path):
    """
    Reads all temporary CSV files in experiment_root/temp_results/,
    merges them with any existing results.csv, sorts them, and deletes the temp dir.
    """
    temp_dir = experiment_root / "temp_results"
    output_path = experiment_root / "results.csv"

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
    if temp_dir.is_dir():
        _load_temp_results(temp_dir, all_rows)

    if not all_rows:
        return

    # Sort by run_id, then regime
    all_rows.sort(key=lambda x: (x["run_id"], x["regime"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(all_rows)

    # Cleanup temp dir if it exists
    if temp_dir.is_dir():
        shutil.rmtree(temp_dir)


def _load_temp_results(temp_dir: Path, all_rows: list):
    """Loads temporary CSV files and appends new rows to all_rows."""
    for file_path in temp_dir.glob("run_*.csv"):
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _coerce_row_types(row)
                # Avoid duplicates if we somehow re-ran a run that was already in results.csv
                if not any(r["run_id"] == row["run_id"] and r["regime"] == row["regime"] for r in all_rows):
                    all_rows.append(row)


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
