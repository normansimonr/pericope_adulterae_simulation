import time
from collections import Counter

import numpy as np


def compute_majority_text_old(genomes):
    genome_list = list(genomes)
    if not genome_list:
        return []
    text_length = len(genome_list[0])
    majority_segments = []
    collation = np.vstack(genome_list)
    for i in range(text_length):
        segment_readings = collation[:, i]
        counts = Counter(segment_readings)
        max_freq = max(counts.values())
        modes = [reading for reading, freq in counts.items() if freq == max_freq]
        majority_segments.append(int(min(modes)))
    return majority_segments


def compute_majority_text_new(genomes):
    genome_list = list(genomes)
    if not genome_list:
        return []
    collation = np.vstack(genome_list)
    num_witnesses, text_length = collation.shape
    majority_segments = []

    # Since our values are small integers [0, 5], we can use bincount or just find modes efficiently
    for i in range(text_length):
        column = collation[:, i]
        # bincount is very fast for small non-negative integers
        counts = np.bincount(column)
        max_freq = np.max(counts)
        # Find all indices that have the max frequency
        modes = np.where(counts == max_freq)[0]
        # Tie-break: min(modes) is already sorted if we use np.where
        majority_segments.append(int(modes[0]))
    return majority_segments


# Benchmarking
n_witnesses = 6000
text_length = 500
genomes = [np.random.randint(0, 6, text_length) for _ in range(n_witnesses)]

start = time.time()
res_old = compute_majority_text_old(genomes)
end = time.time()
print(f"Old method: {end - start:.4f}s")

start = time.time()
res_new = compute_majority_text_new(genomes)
end = time.time()
print(f"New method: {end - start:.4f}s")

assert res_old == res_new
print("Results match!")
