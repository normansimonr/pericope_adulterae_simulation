"""
Provides a centralized, reproducible random number generation (RNG) factory.

This module is designed to solve the critical problem of managing randomness
in a scientific simulation, especially in a parallel execution environment.
Using a single, user-defined integer seed for an entire batch of simulations,
it produces a deterministic and independent stream of random numbers for each
individual simulation run.

How it works:
1.  A top-level `RNGContext` is created from a single integer 'batch seed'.
2.  This context initializes a root `numpy.random.SeedSequence`.
3.  The `spawn()` method uses the root sequence to create `n` independent
    child `SeedSequence` objects.
4.  Each child sequence is used to initialize a `numpy.random.Generator`, which
    is the modern NumPy API for RNG.

This approach guarantees that:
-   The entire batch of simulations is reproducible from a single seed.
-   Each simulation run receives its own independent RNG, preventing state
    contention in parallel or concurrent execution scenarios.
-   The results will be identical regardless of execution order or degree of
    parallelism, because the sequence of generators is determined solely by the
    initial seed.

Users should never use `numpy.random` directly, but instead receive a generator
produced by this factory as part of their simulation's context.
"""

from typing import List, Optional

import numpy as np


class RNGContext:
    """
    Represents the root randomness context for a batch of simulations.

    An `RNGContext` is initialized from a single integer seed. It serves as a
    factory for creating a deterministic sequence of independent random number
    generators for individual simulation runs.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the context from a user-provided seed.

        If no seed is provided, a default, fixed seed is used to ensure
        reproducibility even when a seed is not explicitly set.

        Args:
            seed: The master seed for the entire simulation batch.
        """
        # A fixed default seed ensures that runs are reproducible by default
        # if the user does not specify a seed.
        effective_seed = seed if seed is not None else 20240105
        self._root_sequence = np.random.SeedSequence(effective_seed)

    def spawn(self, n: int) -> List[np.random.Generator]:
        """
        Spawns a fixed number of independent, reproducible child generators.

        For a given `RNGContext` instance, calling this method with the same `n`
        will always produce the exact same list of generators, ensuring
        determinism. Each generator is guaranteed to be independent, making it
        safe for use in parallel simulation runs.

        Args:
            n: The number of child generators to spawn.

        Returns:
            A list containing `n` independent NumPy random number generators.
        """
        child_sequences = self._root_sequence.spawn(n)
        return [np.random.default_rng(seq) for seq in child_sequences]
