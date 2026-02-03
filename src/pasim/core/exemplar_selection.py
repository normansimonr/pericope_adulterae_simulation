"""
Exemplar Selection Logic for Manuscript Generation.

This module provides a pure function, `select_exemplars`, that implements the
core logic for choosing parent exemplars for a newly spawned manuscript. The
selection process is a two-stage filter that respects both geographical
proximity and textual authority (reputation).

The process is as follows:
1.  **Geographical Filtering**: First, a candidate pool of existing manuscripts
    is formed by considering only those in the same geographical region as the
    new manuscript. From this pool, the 10 closest manuscripts (by Euclidean
    distance) are selected. This prioritizes local transmission.

2.  **Mapping to Witness Instances**: The candidate manuscripts are then mapped
    to their corresponding witness instances in the genealogy graph. These
    witness instances, not the manuscripts themselves, are the actual exemplars.

3.  **Reputation-based Ranking**: The candidate witness instances are sorted by
    their `reputation` score in descending order. This ensures that more
    authoritative texts are preferred.

4.  **Final Selection**: A random number of exemplars (`n`, typically 1) is
    drawn from a distribution, and the top `n` witness instances from the
    reputation-sorted list are chosen.

This design cleanly separates manuscript-level properties (location, region)
from witness-instance-level properties (reputation), ensuring that geographical
constraints are applied before textual authority is considered.
"""

import math
from typing import Any, Dict, List

from numpy.random import Generator as RNG

from pasim.core.genealogy import GenealogyGraph
from pasim.core.state import Manuscript

# Type alias for a witness instance ID
WitnessInstanceID = Any


def _euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def select_exemplars(
    new_manuscript: Manuscript,
    alive_manuscripts_in_region: List[Manuscript],
    graph: GenealogyGraph,
    manuscript_to_instance_map: Dict[str, WitnessInstanceID],
    rng: RNG,
) -> List[WitnessInstanceID]:
    """
    Selects parent exemplars for a new manuscript.

    Args:
        new_manuscript: The newly spawned manuscript object.
        alive_manuscripts_in_region: A list of all currently alive manuscripts
                                     in the same region as the new one.
        graph: The full genealogy graph.
        manuscript_to_instance_map: A mapping from manuscript IDs to their
                                    corresponding witness instance IDs in the graph.
        rng: The seeded random number generator.

    Returns:
        A list of witness instance IDs chosen as exemplars.
    """
    if not alive_manuscripts_in_region:
        return []

    # 1. Geographical Filtering: Find 10 closest manuscripts
    distances = [
        (_euclidean_distance(new_manuscript.location, ms.location), ms)
        for ms in alive_manuscripts_in_region
    ]
    distances.sort(key=lambda x: x[0])
    closest_manuscripts = [ms for _, ms in distances[:10]]

    # 2. Map manuscripts to witness instances
    candidate_instances = [
        manuscript_to_instance_map[ms.manuscript_id] for ms in closest_manuscripts
    ]

    # 3. Reputation-based Ranking
    candidate_instances.sort(
        key=lambda inst_id: graph.nodes[inst_id]["reputation"], reverse=True
    )

    # 4. Determine number of exemplars
    n = rng.choice([1, 2, 3], p=[0.8, 0.1, 0.1])
    n = min(n, len(candidate_instances))

    # 5. Final Selection
    return candidate_instances[:n]
