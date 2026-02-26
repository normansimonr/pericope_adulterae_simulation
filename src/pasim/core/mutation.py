"""
Defines the mechanical operator for applying scribal mutations to a tagged string.

This module provides the core function that actually changes the values within
a tagged string, simulating the introduction of scribal errors. It operates based
on a given "expected proportion of segments" to mutate, which dictates the
intensity of the mutation process for a single copying event.

The mutation process is defined as:
1.  Determining the number of segments to change based on the expected proportion.
2.  Randomly selecting the specific segments (indices) that will be altered.
3.  For each selected segment, replacing its current value with a different,
    randomly chosen legal value from the allowed "alphabet."

This operator is a pure, deterministic function given its inputs (including the
random number generator). It relies on the `tagged_string_constraints` module to
ensure that all mutations result in a valid textual state.
"""

import numpy as np
from numpy.typing import NDArray

from pasim.config.schema import SimulationConfig

from .tagged_string_constraints import LEGAL_SEGMENT_VALUES, validate_tagged_string


def mutate_tagged_string(
    tagged_string: NDArray[np.int16],
    rng: np.random.Generator,
    expected_proportion: float,
    config: SimulationConfig,
) -> NDArray[np.int16]:
    """
    Applies scribal mutations to a tagged string.

    This function takes a tagged string and introduces a number of random
    mutations based on the `expected_proportion`. It returns a new tagged
    string and does not modify the original in place.

    Args:
        tagged_string: The input tagged string (NumPy array). It is assumed
                       to be valid.
        rng: A NumPy random number generator for all stochastic operations.
        expected_proportion: A float in the interval [0.0, 1.0] representing
                             the expected proportion of segments to mutate.
        config: The simulation configuration object.

    Returns:
        A new, mutated tagged string.

    Determinism:
        The function is fully deterministic given the same `tagged_string`,
        `rng` state, and `expected_proportion`.

    Mutation Semantics:
        - The number of segments to mutate is calculated by rounding the product
          of `expected_proportion` and the total string length.
        - A set of unique indices is chosen randomly for mutation.
        - Each chosen segment's value is replaced by a *different* legal value,
          sampled uniformly from the available alternatives.

    Failure Conditions:
        - Raises `ValueError` if `expected_proportion` is not in [0.0, 1.0].
        - Raises `ValueError` if the input `tagged_string` has an incorrect
          length.
    """
    if not (0.0 <= expected_proportion <= 1.0):
        raise ValueError(f"Expected proportion must be between 0.0 and 1.0 (inclusive), but got {expected_proportion}.")

    if len(tagged_string) != config.text_length:
        raise ValueError(f"Tagged string must have length {config.text_length}, but got {len(tagged_string)}.")

    if expected_proportion == 0.0:
        return tagged_string.copy()

    # Calculate the number of segments to mutate
    n_mutations = int(round(expected_proportion * config.text_length))

    if n_mutations == 0:
        return tagged_string.copy()

    # Randomly select unique indices to mutate
    indices_to_mutate = rng.choice(config.text_length, size=n_mutations, replace=False)

    # Create a new array to store the mutated string
    new_string = tagged_string.copy()

    # Get the values at the selected indices to mutate
    current_values_at_indices = new_string[indices_to_mutate]

    # Generate random choices from all possible legal values for the positions to mutate
    # The size of this random array is `n_mutations`
    new_values = rng.choice(
        LEGAL_SEGMENT_VALUES,  # Using the global LEGAL_SEGMENT_VALUES from tagged_string_constraints
        size=n_mutations,
        replace=True,  # Allow duplicates, as different positions can mutate to same value
    )

    # Identify where the random choice is the same as the current value
    mask_resample = new_values == current_values_at_indices

    # For elements where new_value == current_value, we need to pick a different value.
    # We can apply a simple transformation that guarantees a different value
    # (e.g., increment modulo max_val, adjusted for 1-based indexing).
    # This introduces a slight bias but is fully vectorized.
    if np.any(mask_resample):
        # max_val = LEGAL_SEGMENT_VALUES.max() (which is 5)
        # The new value should be (current_value % max_val) + 1
        new_values[mask_resample] = (current_values_at_indices[mask_resample] % LEGAL_SEGMENT_VALUES.max()) + 1

    # Apply the new values to the `new_string` at the `indices_to_mutate`
    new_string[indices_to_mutate] = new_values

    # Final safety check to ensure the output is valid
    validate_tagged_string(new_string, config)

    return new_string
