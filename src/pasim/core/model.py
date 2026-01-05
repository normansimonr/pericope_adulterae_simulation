"""
Defines the core data structure for textual state and its factory functions.

In this simulation, the textual state of a witness instance is not represented
as a string of characters, but as a "tagged string." This is a NumPy array of
fixed length containing integers. Each element of the array represents a
textual 'segment', and the integer value of that segment corresponds to a
specific 'reading' (i.e., a textual variant) at that position. Crucially,
this current model focuses solely on the segments and their readings, and
does not include any associated metadata tags for the overall tagged string.
This abstraction allows for efficient, language-agnostic comparison and mutation.

This module provides a factory layer for creating, copying, and mutating
these tagged strings. Using factory functions instead of ad hoc array creation
ensures that all tagged strings are constructed and modified in a consistent,
reproducible, and deterministic manner. All randomness is explicitly injected
from a `numpy.random.Generator`, making the entire process controllable and
independent of global state.
"""

import numpy as np
from numpy.typing import NDArray

# The fixed length of every tagged string in the simulation.
TAGGED_STRING_LENGTH = 100

# The data type for the integer readings in each segment. np.int16 is chosen
# as a balance between memory efficiency and range. It provides a range of
# -32,768 to 32,767, which is more than sufficient for representing all
# conceivable textual variants or states at any given position in the text.
TAGGED_STRING_DTYPE = np.int16


def create_tagged_string(rng: np.random.Generator) -> NDArray[np.int16]:
    """
    Creates a new, randomly initialized tagged string.

    The creation is deterministic; given the same RNG state, this function
    will always produce the identical tagged string.

    Args:
        rng: A NumPy random number generator for reproducible randomness.

    Returns:
        A new NumPy array representing a tagged string.
    """
    # For now, we initialize with small integers. The range can be adjusted
    # to match the expected number of variants in the model.
    return rng.integers(
        low=0,
        high=1000,
        size=TAGGED_STRING_LENGTH,
        dtype=TAGGED_STRING_DTYPE
    )


def copy_tagged_string(tagged_string: NDArray[np.int16]) -> NDArray[np.int16]:
    """
    Creates a safe, deep copy of a tagged string.

    This function always returns a new NumPy array, not a view, ensuring that
    the original tagged string cannot be mutated by downstream modifications
    to the copy.

    Args:
        tagged_string: The tagged string to copy.

    Returns:
        A new NumPy array with the same shape and data.
    """
    assert tagged_string.shape == (TAGGED_STRING_LENGTH,), "Invalid tagged string shape"
    assert tagged_string.dtype == TAGGED_STRING_DTYPE, "Invalid tagged string dtype"
    return tagged_string.copy()


def mutate_tagged_string(
    tagged_string: NDArray[np.int16], rng: np.random.Generator
) -> NDArray[np.int16]:
    """
    Applies a single, random mutation to a tagged string.

    This function is deterministic based on the input RNG state. It returns a
    new, mutated tagged string and does not modify the original array in place.

    The current mutation logic is a simple placeholder: it picks one random
    segment in the array and changes its value to a new random reading.

    Args:
        tagged_string: The original tagged string to mutate.
        rng: A NumPy random number generator for reproducible mutation.

    Returns:
        A new tagged string with a single mutation applied.
    """
    new_string = copy_tagged_string(tagged_string)

    # Pick a random segment and a new random value for the reading.
    index_to_mutate = rng.integers(0, TAGGED_STRING_LENGTH)
    new_value = rng.integers(0, 1000, dtype=TAGGED_STRING_DTYPE)

    new_string[index_to_mutate] = new_value
    return new_string