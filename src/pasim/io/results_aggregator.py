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

    for file_path in temp_dir.glob("run_*.csv"):
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields back to their proper types so they are not quoted by QUOTE_NONNUMERIC
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
                    if row[field] != "":
                        row[field] = float(row[field])

                all_rows.append(row)

    # Sort by run_id, then regime
    all_rows.sort(key=lambda x: (x["run_id"], x["regime"]))

    output_path = experiment_root / "results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(all_rows)

    # Cleanup
    shutil.rmtree(temp_dir)
