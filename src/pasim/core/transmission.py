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


def majority_from_exemplars(parent_texts: List[TaggedString], rng: np.random.Generator) -> TaggedString:
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
    # num_segments = first_text.shape[0]  # Assuming 1D tagged strings
    dtype = first_text.dtype

    if not all(p.shape == first_text.shape and p.dtype == dtype for p in parent_texts):
        raise ValueError("All parent exemplars must have the same shape and dtype.")

    # Stack arrays for efficient, column-wise processing
    # stacked_texts will have shape (num_parents, num_segments)
    stacked_texts = np.array(parent_texts)  # Automatically stacks if List[NDArray]

    # Assume legal segment values are 1, 2, 3, 4, 5 as per tagged_string_constraints
    possible_values = np.array([1, 2, 3, 4, 5], dtype=dtype)
    # num_possible_values = len(possible_values) # Not directly used in new vectorized code

    # Create a mask for each possible value across all parents and segments
    # expanded_values shape: (num_possible_values, 1, 1)
    # stacked_texts shape: (1, num_parents, num_segments)
    # comparison result shape: (num_possible_values, num_parents, num_segments)
    matches_per_value_parent_segment = possible_values[:, np.newaxis, np.newaxis] == stacked_texts[np.newaxis, :, :]

    # Count occurrences of each value (1-5) for each segment
    # Sum along num_parents axis (axis=1): (num_possible_values, num_segments)
    counts_per_value_segment = np.sum(matches_per_value_parent_segment, axis=1)

    # Find the maximum count for each segment
    # max_counts_per_segment shape: (num_segments,)
    max_counts_per_segment = np.max(counts_per_value_segment, axis=0)

    # Identify which values (1-5) achieve this maximum count for each segment
    # tied_mask shape: (num_possible_values, num_segments)
    tied_mask = counts_per_value_segment == max_counts_per_segment[np.newaxis, :]

    # Handle ties by random choice:
    # Generate random numbers for each potential choice where there's a tie
    random_choices = rng.random(tied_mask.shape)
    # For values that are not tied for the max count, set their random choice score to a very low number
    # This ensures they won't be picked by argmax
    random_choices[~tied_mask] = -1

    # Find the index of the best random choice for each segment.
    # np.argmax will pick the first if random scores are equal for tied elements.
    chosen_value_indices = np.argmax(random_choices, axis=0)

    # Map these indices back to the actual possible values
    result_text = possible_values[chosen_value_indices]

    return result_text
