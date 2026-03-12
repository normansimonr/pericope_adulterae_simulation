import csv
from pathlib import Path
from typing import List, Optional


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
    pct_sampled_witnesses_with_pa: Optional[float] = None,
    pct_majority_disagree_autograph: Optional[float] = None,
    pct_all_witnesses_with_pa: Optional[float] = None,
    ideal_majority_text_segments: Optional[List[int]] = None,
    pct_ideal_majority_disagree_autograph: Optional[float] = None,
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
    ideal_majority_text_str = serialize_majority_text(ideal_majority_text_segments) if ideal_majority_text_segments is not None else ""

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

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "run_seed": run_seed,
            "regime": regime,
            "total_manuscripts_spawned": total_manuscripts_spawned,
            "majority_text": majority_text_str,
            "pct_sampled_witnesses_with_pa": pct_sampled_witnesses_with_pa if pct_sampled_witnesses_with_pa is not None else "",
            "pct_majority_disagree_autograph": pct_majority_disagree_autograph if pct_majority_disagree_autograph is not None else "",
            "pct_all_witnesses_with_pa": pct_all_witnesses_with_pa if pct_all_witnesses_with_pa is not None else "",
            "ideal_majority_text": ideal_majority_text_str,
            "pct_ideal_majority_disagree_autograph": pct_ideal_majority_disagree_autograph
            if pct_ideal_majority_disagree_autograph is not None
            else "",
        })
