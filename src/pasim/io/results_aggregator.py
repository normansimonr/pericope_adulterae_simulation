import csv
import shutil
from pathlib import Path


def aggregate_results(experiment_root: Path):
    """
    Reads all temporary CSV files in experiment_root/temp_results/,
    merges them into a single results.csv, sorts them, and deletes the temp dir.
    """
    temp_dir = experiment_root / "temp_results"
    if not temp_dir.is_dir():
        return

    all_rows = []
    fieldnames = ["run_id", "run_seed", "regime", "total_manuscripts_spawned", "majority_text"]

    for file_path in temp_dir.glob("run_*.csv"):
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert run_id to int for proper sorting
                row["run_id"] = int(row["run_id"])
                all_rows.append(row)

    # Sort by run_id, then regime
    all_rows.sort(key=lambda x: (x["run_id"], x["regime"]))

    output_path = experiment_root / "results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Cleanup
    shutil.rmtree(temp_dir)
