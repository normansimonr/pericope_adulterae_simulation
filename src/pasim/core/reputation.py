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

from typing import Dict, Optional
import math
import numpy as np


# Define the default mapping from reputation level to expected mutation proportion.
# Higher reputation (e.g., 5) implies a lower expected mutation proportion.
# These values are placeholders and can be overridden by users.
DEFAULT_REPUTATION_MAPPING: Dict[int, float] = {
    1: 0.10,  # Very low reputation = 10% of segments expected to mutate
    2: 0.10,  # Low reputation = 10%
    3: 0.30,  # Medium reputation = 30%
    4: 0.30,  # High reputation = 30%
    5: 0.20,  # Very high reputation = 20%
}


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


def expected_mutation_proportion(
    reputation: int, mapping: Optional[Dict[int, float]] = None
) -> float:
    """
    Returns the expected proportion of segments that should mutate for a given reputation.

    This function serves as the policy lookup. It takes a witness's reputation
    level and translates it into a quantifiable expectation of error intensity.

    Args:
        reputation: The integer reputation level (1-5) of the witness.
        mapping: An optional dictionary to override the default reputation
                 to proportion mapping. If None, `DEFAULT_REPUTATION_MAPPING`
                 will be used.

    Returns:
        A float representing the expected proportion of segments that will mutate,
        in the range [0.0, 1.0].

    Determinism:
        This function is fully deterministic.

    Failure Conditions:
        - Raises `ValueError` if the provided `reputation` is invalid.
        - Raises `ValueError` if the `reputation` is not found in the chosen mapping.
        - Raises `ValueError` if the mapped proportion is not within [0.0, 1.0].
    """
    validate_reputation(reputation)

    active_mapping = mapping if mapping is not None else DEFAULT_REPUTATION_MAPPING

    if reputation not in active_mapping:
        raise ValueError(f"Reputation level '{reputation}' not found in the mapping.")

    proportion = active_mapping[reputation]

    if not (0.0 <= proportion <= 1.0):
        raise ValueError(
            f"Mapped mutation proportion '{proportion}' for reputation '{reputation}' "
            "is not within the valid range [0.0, 1.0]."
        )

    return proportion


def sample_reputation(
    rng: np.random.Generator,
    reputation_distribution: Dict[int, float]
) -> int:
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
