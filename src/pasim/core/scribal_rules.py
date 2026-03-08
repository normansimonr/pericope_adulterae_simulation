"""
Implements the composite scribal rule, combining base transmission and mutation.

This module provides the high-level function that simulates the complete act of
a scribe copying a text. It orchestrates the previously defined, lower-level
primitives into a single, coherent pipeline. This represents the core of the
textual evolution model in the simulation.

The scribal pipeline proceeds in three distinct stages:
1.  **Base Transmission**: First, a "clean" base text is produced from one or
    more parent exemplars. If there is only one exemplar, this is a direct
    copy. If there are multiple (a "contamination" scenario), their texts are
    combined via segment-wise majority voting to produce a single resolved text.
2.  **Error Intensity Determination**: The reputation of the witness instance
    being created is used to determine the expected proportion of segments that
    will be affected by scribal error. This is a policy decision that maps an
    abstract quality (reputation) to a concrete error rate.
3.  **Mutation**: Finally, the mutation operator is applied to the base text,
    using the determined error intensity to introduce a specific number of
    random changes.

The result is a new tagged string, representing the text of the newly created
witness instance, which has been influenced by both its ancestry (contamination)
and the scribal process itself (error).
"""

from typing import Dict, List, Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pasim.config.schema import SimulationConfig
from pasim.core.mutation import mutate_tagged_string
from pasim.core.reputation import expected_mutation_proportion
from pasim.core.transmission import copy_from_single_exemplar, majority_from_exemplars

# Define a type alias for a tagged string (a NumPy array of int16)
TaggedString: TypeAlias = NDArray[np.int16]


def apply_scribal_rule(
    exemplar_texts: List[TaggedString],
    rng: np.random.Generator,
    reputation: int,
    config: SimulationConfig,
    mutation_mapping: Optional[Dict[int, float]] = None,
) -> TaggedString:
    """
    Produces a new, copied and mutated tagged string from one or more exemplars.

    This function is the main entry point for the scribal error pipeline. It
    composes base transmission, reputation-based error intensity, and the
    mutation operator to generate the text for a new witness instance.

    Args:
        exemplar_texts: A list containing one or more tagged strings to be used
                        as exemplars for the new copy.
        rng: A NumPy random number generator for all stochastic operations
             (tie-breaking in majority voting and all mutation mechanics).
        reputation: The integer reputation level (1-5) of the scribe/witness,
                    which determines the error intensity.
        config: The simulation configuration object.
        mutation_mapping: An optional dictionary to override the default
                          reputation-to-mutation-proportion mapping.

    Returns:
        A new tagged string representing the final, potentially mutated text.

    Determinism:
        This function is fully deterministic given the same inputs and the same
        `rng` state.

    Failure Conditions:
        - Raises `ValueError` if `exemplar_texts` is empty.
        - Raises `ValueError` if the provided `reputation` is invalid.
        - Raises `ValueError` if the exemplar texts have inconsistent shapes
          or dtypes (as enforced by the underlying transmission functions).
    """
    if not exemplar_texts:
        raise ValueError("Cannot apply scribal rule without at least one exemplar.")

    # Stage 1: Base Transmission
    if len(exemplar_texts) == 1:
        base_text = copy_from_single_exemplar(exemplar_texts[0])
    else:
        base_text = majority_from_exemplars(exemplar_texts, rng)

    # Stage 2: Error Intensity Determination
    proportion = expected_mutation_proportion(reputation, mutation_mapping)

    # Stage 3: Mutation
    # RULE: If a manuscript has exactly one parent and the inherited reading for
    # a segment is 0, then mutation must not occur at that segment.
    immutable_mask = None
    if len(exemplar_texts) == 1:
        immutable_mask = (base_text == 0)

    mutated_text = mutate_tagged_string(base_text, rng, proportion, config, immutable_mask=immutable_mask)

    return mutated_text
