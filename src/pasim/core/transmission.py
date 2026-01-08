"""
Defines the base rules for textual transmission from exemplars to a new copy.

This module provides deterministic, structural rules for how a new tagged string
is generated from one or more parent (exemplar) tagged strings. This represents
the "ideal" copying process, completely isolated from any subsequent scribal
errors, mutations, or other stochastic alterations.

The concept of "base transmission" is crucial because it separates the act of
combining sources from the act of introducing errors. By having this clean
base layer, the simulation can later apply various scribal error models on top
of a consistent, predictable foundation. This ensures that the logic for source
amalgamation (e.g., majority voting) is distinct and testable in isolation
from the logic of textual corruption.
"""

from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

# For type hinting clarity
TaggedString = NDArray[np.int16]


def copy_from_single_exemplar(parent_text: TaggedString) -> TaggedString:
    """
    Produces an exact, deep copy of a single exemplar's tagged string.

    This represents the simplest form of transmission: a scribe has one source
    and creates a perfect replica. The function returns a new NumPy array, not
    a view, ensuring the copy is independent of the original.

    Args:
        parent_text: The tagged string of the single parent exemplar.

    Returns:
        A new NumPy array with the exact same shape, dtype, and data as the
        input array.

    Determinism:
        This function is fully deterministic.
    """
    return parent_text.copy()


def majority_from_exemplars(
    parent_texts: List[TaggedString], rng: np.random.Generator
) -> TaggedString:
    """
    Combines multiple exemplar tagged strings via segment-wise majority voting.

    For each segment in the tagged string, this function determines the most
    common value ("reading") among all parent exemplars. If there is a single
    most common value, it is chosen for the new text. If there is a tie for the
    most common value, one of the tied values is chosen randomly.

    Args:
        parent_texts: A list of tagged strings (NumPy arrays) to be used as
                      exemplars.
        rng: A NumPy random number generator, used exclusively for tie-breaking.

    Returns:
        A new tagged string representing the majority text.

    Determinism:
        This function is deterministic given the same set of `parent_texts` and
        the same `rng` state.

    Failure Conditions:
        - Raises `ValueError` if fewer than 2 parent texts are provided.
        - Raises `ValueError` if the parent texts do not all have the same
          shape and dtype.
    """
    if len(parent_texts) < 2:
        raise ValueError("Majority voting requires at least two parent exemplars.")

    first_text = parent_texts[0]
    shape = first_text.shape
    dtype = first_text.dtype

    if not all(p.shape == shape and p.dtype == dtype for p in parent_texts):
        raise ValueError("All parent exemplars must have the same shape and dtype.")

    # Stack arrays for efficient, column-wise processing
    stacked_texts = np.vstack(parent_texts)
    num_segments = stacked_texts.shape[1]
    result_text = np.empty(num_segments, dtype=dtype)

    for i in range(num_segments):
        segment_values = stacked_texts[:, i]
        values, counts = np.unique(segment_values, return_counts=True)

        max_count = np.max(counts)
        # Find all values that share the maximum count
        tied_values = values[counts == max_count]

        if len(tied_values) == 1:
            # Strict majority, no tie
            result_text[i] = tied_values[0]
        else:
            # Tie for majority, break it randomly
            result_text[i] = rng.choice(tied_values)

    return result_text
