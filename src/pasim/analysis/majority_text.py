import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import numpy as np


def compute_majority_text(genomes: Iterable[np.ndarray]) -> List[int]:
    """
    Computes the majority text from an iterable of genomes.
    The majority text is the mode reading per segment, with the smallest value
    chosen in case of a tie.
    """
    # Convert iterable to a list to check its length and handle the empty case
    genome_list = list(genomes)
    if not genome_list:
        return []

    # All genomes are assumed to have the same length
    text_length = len(genome_list[0])
    majority_segments = []

    # Stack the genomes into a 2D array: rows = witnesses, columns = segments
    # This allows efficient segment-wise processing.
    collation = np.vstack(genome_list)

    for i in range(text_length):
        segment_readings = collation[:, i]
        # Counter provides counts of each reading
        counts = Counter(segment_readings)
        # Find the maximum frequency
        max_freq = max(counts.values())
        # Find all readings that share the maximum frequency
        modes = [reading for reading, freq in counts.items() if freq == max_freq]
        # Tie-break: choose the smallest value
        majority_segments.append(int(min(modes)))

    return majority_segments


def save_majority_text(majority_segments: List[int], regime_dir: Path):
    """
    Saves the computed majority text to a JSON file in the regime directory.
    """
    output_path = regime_dir / "majority_text.json"
    data = {"segment_count": len(majority_segments), "majority_segments": majority_segments}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
