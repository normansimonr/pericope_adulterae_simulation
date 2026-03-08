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
from typing import Optional
from numpy.typing import NDArray

from pasim.config.schema import SimulationConfig

from .tagged_string_constraints import LEGAL_SEGMENT_VALUES, validate_tagged_string


def mutate_tagged_string(
    tagged_string: NDArray[np.int16],
    rng: np.random.Generator,
    expected_proportion: float,
    config: SimulationConfig,
    immutable_mask: Optional[NDArray[np.bool_]] = None,
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
        immutable_mask: An optional boolean mask where True indicates segments
                         that must NOT be mutated.

    Returns:
        A new, mutated tagged string.

    Determinism:
        The function is fully deterministic given the same `tagged_string`,
        `rng` state, and `expected_proportion`.

    Mutation Semantics:
        - The number of segments to mutate is calculated by rounding the product
          of `expected_proportion` and the total string length.
        - A set of unique indices is chosen randomly for mutation.
        - If an `immutable_mask` is provided, indices are chosen only from
          the mutable segments (where the mask is False).
        - If the number of requested mutations exceeds the available mutable
          segments, all available mutable segments are mutated.
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

    # Calculate the target number of segments to mutate
    n_mutations = int(round(expected_proportion * config.text_length))

    if n_mutations == 0:
        return tagged_string.copy()

    # Identify indices that can be mutated
    if immutable_mask is not None:
        mutable_indices = np.where(~immutable_mask)[0]
    else:
        mutable_indices = np.arange(config.text_length)

    # Adjust n_mutations if it exceeds available mutable indices
    n_to_mutate = min(n_mutations, len(mutable_indices))

    if n_to_mutate == 0:
        return tagged_string.copy()

    # Randomly select unique indices to mutate from the mutable set
    indices_to_mutate = rng.choice(mutable_indices, size=n_to_mutate, replace=False)

    # Create a new array to store the mutated string
    new_string = tagged_string.copy()

    # Get the values at the selected indices to mutate
    current_values_at_indices = new_string[indices_to_mutate]

    # Generate random choices from all possible legal values for the positions to mutate
    # The size of this random array is `n_to_mutate`
    new_values = rng.choice(
        LEGAL_SEGMENT_VALUES,  # Using the global LEGAL_SEGMENT_VALUES from tagged_string_constraints
        size=n_to_mutate,
        replace=True,  # Allow duplicates, as different positions can mutate to same value
    )

    # Identify where the random choice is the same as the current value
    mask_resample = new_values == current_values_at_indices

    # If some elements picked the same value as their current value, we need to pick
    # a different value. To remain deterministic and vectorized, we shift to the
    # "next" value in the sorted alphabet.
    if np.any(mask_resample):
        alphabet_size = len(LEGAL_SEGMENT_VALUES)
        # Find the index of each current value in the sorted alphabet.
        # np.searchsorted is efficient for this.
        indices_in_alphabet = np.searchsorted(LEGAL_SEGMENT_VALUES, current_values_at_indices[mask_resample])

        # Shift the index by 1 (wrapping around) to guarantee a different value.
        new_alphabet_indices = (indices_in_alphabet + 1) % alphabet_size

        # Map back to the actual legal values.
        new_values[mask_resample] = LEGAL_SEGMENT_VALUES[new_alphabet_indices]

    # Apply the new values to the `new_string` at the `indices_to_mutate`
    new_string[indices_to_mutate] = new_values

    # Final safety check to ensure the output is valid
    validate_tagged_string(new_string, config)

    return new_string
