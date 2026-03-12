import json
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

    # Stack the genomes into a 2D array: rows = witnesses, columns = segments
    # This allows efficient segment-wise processing.
    collation = np.vstack(genome_list)
    majority_segments = []

    # Optimized segment-wise mode calculation using np.bincount
    # np.bincount is very fast for small non-negative integers.
    # Our segments use [0, 5], fitting this perfectly.
    for i in range(text_length):
        column = collation[:, i]
        counts = np.bincount(column)
        max_freq = np.max(counts)
        # Find all indices that share the max frequency
        modes = np.where(counts == max_freq)[0]
        # Tie-break: min(modes) is the first index in the returned array
        majority_segments.append(int(modes[0]))

    return majority_segments


def save_majority_text(majority_segments: List[int], regime_dir: Path):
    """
    Saves the computed majority text to a JSON file in the regime directory.
    """
    output_path = regime_dir / "majority_text.json"
    data = {"segment_count": len(majority_segments), "majority_segments": majority_segments}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
