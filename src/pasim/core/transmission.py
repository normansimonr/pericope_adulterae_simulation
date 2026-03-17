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

from typing import List, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pasim.core.tagged_string_constraints import LEGAL_SEGMENT_VALUES

# For type hinting clarity
TaggedString: TypeAlias = NDArray[np.int16]


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


def majority_from_exemplars(parent_texts: List[TaggedString], parent_reputations: List[int], rng: np.random.Generator) -> TaggedString:
    """
    Combines multiple exemplar tagged strings via weighted segment-wise majority voting.

    For each segment in the tagged string, this function determines the "winning"
    reading among all parent exemplars. Instead of a simple count, each parent's
    contribution is weighted by its reputation score (1-5). If there is a tie
    for the highest weighted sum, one of the tied values is chosen randomly.

    Args:
        parent_texts: A list of tagged strings (NumPy arrays) to be used as
                      exemplars.
        parent_reputations: A list of integer reputation scores (1-5) corresponding
                            to each parent text.
        rng: A NumPy random number generator, used exclusively for tie-breaking.

    Returns:
        A new tagged string representing the weighted majority text.

    Determinism:
        This function is deterministic given the same inputs and rng state.

    Failure Conditions:
        - Raises `ValueError` if fewer than 2 parent texts are provided.
        - Raises `ValueError` if the number of reputations doesn't match the number of texts.
        - Raises `ValueError` if the parent texts do not all have the same
          shape and dtype.
    """
    if len(parent_texts) < 2:
        raise ValueError("Majority voting requires at least two parent exemplars.")

    if len(parent_texts) != len(parent_reputations):
        raise ValueError("The number of parent reputations must match the number of parent texts.")

    first_text = parent_texts[0]
    dtype = first_text.dtype

    if not all(p.shape == first_text.shape and p.dtype == dtype for p in parent_texts):
        raise ValueError("All parent exemplars must have the same shape and dtype.")

    # Stack arrays for efficient processing: (num_parents, num_segments)
    stacked_texts = np.array(parent_texts)
    weights = np.array(parent_reputations, dtype=float)

    # Use the authoritative legal segment values from tagged_string_constraints
    possible_values = LEGAL_SEGMENT_VALUES.astype(dtype)

    # matches_per_value_parent_segment shape: (num_possible_values, num_parents, num_segments)
    matches_per_value_parent_segment = possible_values[:, np.newaxis, np.newaxis] == stacked_texts[np.newaxis, :, :]

    # Calculate weighted sums for each value and segment.
    # We multiply the boolean matches (broadcasted weights) and sum along num_parents axis.
    # weighted_sums_per_value_segment shape: (num_possible_values, num_segments)
    weighted_sums_per_value_segment = np.sum(matches_per_value_parent_segment * weights[np.newaxis, :, np.newaxis], axis=1)

    # Find the maximum weighted sum for each segment
    max_sums_per_segment = np.max(weighted_sums_per_value_segment, axis=0)

    # Identify which values achieve this maximum sum
    tied_mask = weighted_sums_per_value_segment == max_sums_per_segment[np.newaxis, :]

    # Handle ties by random choice:
    random_choices = rng.random(tied_mask.shape)
    random_choices[~tied_mask] = -1

    chosen_value_indices = np.argmax(random_choices, axis=0)
    result_text = possible_values[chosen_value_indices]

    return result_text
