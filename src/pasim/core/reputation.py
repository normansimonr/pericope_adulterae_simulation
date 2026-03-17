"""
Maps a witness instance's reputation level to an expected error intensity.

This module establishes a policy layer that translates an abstract "reputation"
score into a concrete, expected proportion of segments that are likely to be
affected by scribal errors during a copying event. It explicitly separates the
policy definition (what a reputation score means in terms of error rate) from
the actual mechanics of introducing those errors into a tagged string.

The "expected proportion of segments" refers to a float value between 0.0 and
1.0, indicating the fraction of segments in a tagged string that are expected
to undergo some form of mutation or error. This provides a configurable,
deterministic intensity level that mutation operators can then consume.
"""

from typing import Dict

import numpy as np


def validate_reputation(reputation: int) -> None:
    """
    Validates that a given reputation level is within the legal range.

    Args:
        reputation: The integer reputation level to validate.

    Returns:
        None, if the reputation is valid.

    Failure Conditions:
        - Raises `ValueError` if the reputation is not an integer between 1 and 5.
    """
    if not isinstance(reputation, int):
        raise ValueError(f"Reputation must be an integer, but got {type(reputation)}.")
    if reputation < 1 or reputation > 5:
        raise ValueError(f"Reputation must be an integer between 1 and 5 (inclusive), but got {reputation}.")


def expected_mutation_proportion(reputation: int, mapping: Dict[int, float]) -> float:
    """
    Returns the expected proportion of segments that should mutate for a given reputation.

    This function serves as the policy lookup. It takes a witness's reputation
    level and translates it into a quantifiable expectation of error intensity.

    Args:
        reputation: The integer reputation level (1-5) of the witness.
        mapping: A dictionary mapping reputation (1-5) to proportion.

    Returns:
        A float representing the expected proportion of segments that will mutate,
        in the range [0.0, 1.0].

    Determinism:
        This function is fully deterministic.

    Failure Conditions:
        - Raises `ValueError` if the provided `reputation` is invalid.
        - Raises `ValueError` if the `reputation` is not found in the mapping.
        - Raises `ValueError` if the mapped proportion is not within [0.0, 1.0].
    """
    validate_reputation(reputation)

    if reputation not in mapping:
        raise ValueError(f"Reputation level '{reputation}' not found in the mapping.")

    proportion = mapping[reputation]

    if not (0.0 <= proportion <= 1.0):
        raise ValueError(
            f"Mapped mutation proportion '{proportion}' for reputation '{reputation}' is not within the valid range [0.0, 1.0]."
        )

    return proportion


def sample_reputation(rng: np.random.Generator, reputation_distribution: Dict[int, float]) -> int:
    """
    Samples a reputation value based on a user-defined probability distribution.

    This function allows for controlled, probabilistic assignment of reputation scores
    to new witness instances, making reputation an experimental parameter.
    It expects a pre-validated `reputation_distribution` from the config.

    Args:
        rng: The seeded random number generator to ensure reproducibility.
        reputation_distribution: A dictionary mapping reputation scores (1-5)
                                 to their probabilities.

    Returns:
        An integer representing the sampled reputation score (1-5).
    """
    reputation_scores = list(reputation_distribution.keys())
    probabilities = list(reputation_distribution.values())

    sampled_reputation = rng.choice(reputation_scores, p=probabilities)
    return int(sampled_reputation)


def sample_inherited_reputation(
    parent_reputations: list[int],
    rng: np.random.Generator,
) -> int:
    """
    Samples a child's reputation based on its parents' reputations with drift.

    This function implements reputation inheritance, where a child's reputation
    is biased toward the maximum reputation of its parent exemplars. It applies
    a stochastic "drift" (70% chance to stay the same, 15% to increase by 1,
    15% to decrease by 1), clipped to the [1, 5] range.

    If no parents are provided, this function defaults to the maximum
    reputation (5), assuming it's being called for the autograph/root.

    Args:
        parent_reputations: A list of integer reputation scores from parent exemplars.
        rng: The seeded random number generator.

    Returns:
        An integer representing the inherited (and potentially drifted) reputation.
    """
    if not parent_reputations:
        # Default for autograph/root node
        return 5

    # Inherit from the best available source
    base_rep = max(parent_reputations)

    # Apply stochastic drift (-1, 0, +1)
    drift = rng.choice([-1, 0, 1], p=[0.15, 0.70, 0.15])
    child_rep = base_rep + drift

    # Clip to legal range [1, 5]
    return int(np.clip(child_rep, 1, 5))
