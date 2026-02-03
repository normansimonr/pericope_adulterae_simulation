"""
Defines the legal state space for the values within a tagged string.

This module provides a single, authoritative definition of the "alphabet" from
which the integer values (readings) in a tagged string's segments can be drawn.
It establishes the foundational constraints for the textual state of any given
witness instance.

The primary reason for isolating these constraints is to ensure that all
other components of the simulation—especially tagged string factories and
mutation operators—can rely on a single source of truth for what constitutes a
valid state. By centralizing this logic, we prevent the state space from being
implicitly defined or accidentally violated by different parts of the
simulation. All higher-level logic that creates or modifies tagged strings must
use the primitives in this module to validate their operations.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from pasim.config.schema import SimulationConfig

# The single, authoritative definition of the set of legal integer values
# that a segment in a tagged string can hold. This is the "alphabet" of
# our textual model. It is defined as a NumPy array for efficient lookups.
LEGAL_SEGMENT_VALUES: NDArray[np.int16] = np.array([1, 2, 3, 4, 5], dtype=np.int16)

# For type hinting clarity.
SegmentValue: TypeAlias = np.int16


def is_valid_segment_value(value: SegmentValue) -> bool:
    """
    Checks if a single value is a valid segment value.

    This function is a simple, side-effect-free predicate that checks if the
    given value is present in the set of `LEGAL_SEGMENT_VALUES`.

    Args:
        value: The integer value to check.

    Returns:
        `True` if the value is a legal segment value, `False` otherwise.
    """
    return value in LEGAL_SEGMENT_VALUES


def validate_tagged_string(
    tagged_string: NDArray[np.int16], config: SimulationConfig
) -> None:
    """
    Verifies that an entire tagged string is structurally and legally valid.

    This function performs a series of checks to ensure the tagged string
    conforms to the simulation's foundational rules. It is intended to be used
    as a safeguard in factory functions and after mutation operations.

    Args:
        tagged_string: The NumPy array representing the tagged string.
        config: The simulation configuration object.

    Returns:
        `None` if the tagged string is valid.

    Failure Conditions:
        - Raises `TypeError` if `tagged_string` is not a NumPy array.
        - Raises `ValueError` if the tagged string does not have the correct
          length (`config.text_length`).
        - Raises `ValueError` if the array's dtype is not integer-like.
        - Raises `ValueError` if any value in the array is not a legal
          segment value.
    """
    if not isinstance(tagged_string, np.ndarray):
        raise TypeError("Tagged string must be a NumPy array.")

    if len(tagged_string) != config.text_length:
        raise ValueError(
            f"Tagged string must have length {config.text_length}, "
            f"but got {len(tagged_string)}."
        )

    if not np.issubdtype(tagged_string.dtype, np.integer):
        raise ValueError(
            "Tagged string dtype must be an integer type, "
            f"but got {tagged_string.dtype}."
        )

    if not np.all(np.isin(tagged_string, LEGAL_SEGMENT_VALUES)):
        raise ValueError("Tagged string contains illegal segment values.")


def sample_alternative_value(
    current_value: SegmentValue, rng: np.random.Generator
) -> SegmentValue:
    """
    Samples a new legal segment value that is different from the current one.

    This is a key primitive for mutation logic. It ensures that when a segment's
    reading changes, it transitions to a different, valid reading from the
    legal alphabet. The selection is deterministic based on the state of the
    provided random number generator.

    Args:
        current_value: The current value of the segment, which must not be
                       returned.
        rng: A NumPy random number generator for reproducible sampling.

    Returns:
        A new, legal segment value that is different from `current_value`.

    Failure Conditions:
        - Raises `ValueError` if `current_value` is not a legal segment value
          itself, as it would be impossible to choose an alternative.
    """
    if not is_valid_segment_value(current_value):
        raise ValueError(f"'{current_value}' is not a valid segment value.")

    # Get the array of all possible alternative values
    alternatives = LEGAL_SEGMENT_VALUES[LEGAL_SEGMENT_VALUES != current_value]

    # Sample one value from the alternatives
    return rng.choice(alternatives)
