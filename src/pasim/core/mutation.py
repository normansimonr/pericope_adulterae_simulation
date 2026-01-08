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

from .tagged_string_constraints import (
    TAGGED_STRING_LENGTH,
    sample_alternative_value,
    validate_tagged_string,
)


def mutate_tagged_string(
    tagged_string: NDArray[np.int16],
    rng: np.random.Generator,
    expected_proportion: float,
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
        raise ValueError(
            "Expected proportion must be between 0.0 and 1.0 (inclusive), "
            f"but got {expected_proportion}."
        )

    if len(tagged_string) != TAGGED_STRING_LENGTH:
        raise ValueError(
            f"Tagged string must have length {TAGGED_STRING_LENGTH}, "
            f"but got {len(tagged_string)}."
        )

    if expected_proportion == 0.0:
        return tagged_string.copy()

    # Calculate the number of segments to mutate
    n_mutations = int(round(expected_proportion * TAGGED_STRING_LENGTH))

    if n_mutations == 0:
        return tagged_string.copy()

    # Randomly select unique indices to mutate
    indices_to_mutate = rng.choice(
        TAGGED_STRING_LENGTH, size=n_mutations, replace=False
    )

    # Create a new array to store the mutated string
    new_string = tagged_string.copy()

    # Apply mutations
    for index in indices_to_mutate:
        current_value = new_string[index]
        new_value = sample_alternative_value(current_value, rng)
        new_string[index] = new_value

    # Final safety check to ensure the output is valid
    validate_tagged_string(new_string)

    return new_string
