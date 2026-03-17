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

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.random import Generator as RNG
from scipy.spatial import KDTree  # type: ignore

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy import GenealogyGraph
from pasim.core.state import Manuscript

# Type alias for a witness instance ID
WitnessInstanceID = Any


def select_exemplars(
    new_manuscript: Manuscript,
    alive_manuscripts_in_region: List[Manuscript],
    graph: GenealogyGraph,
    manuscript_to_instance_map: Dict[str, WitnessInstanceID],
    rng: RNG,
    config: SimulationConfig,
    kdtree: Optional[KDTree] = None,
    reputation_cache: Optional[Dict[WitnessInstanceID, float]] = None,
    age_cache: Optional[Dict[WitnessInstanceID, int]] = None,
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
        config: The simulation configuration.
        kdtree: An optional pre-built KDTree for the `alive_manuscripts_in_region`.
        reputation_cache: An optional dictionary mapping instance IDs to reputations.
        age_cache: An optional dictionary mapping instance IDs to birth ticks.

    Returns:
        A list of witness instance IDs chosen as exemplars.
    """
    if not alive_manuscripts_in_region:
        return []

    # 1. Geographical Filtering: Find closest manuscripts using KDTree
    # Use provided KDTree or build a new one if not provided
    if kdtree is None:
        locations = np.array([ms.location for ms in alive_manuscripts_in_region])
        tree = KDTree(locations)
    else:
        tree = kdtree

    # Query for nearest neighbors.
    k = min(config.geographical_candidate_pool_size, len(alive_manuscripts_in_region))
    distances, indices = tree.query(new_manuscript.location, k=k)

    # Handle the case where k=1, which returns a single index instead of an array
    if k == 1 and np.isscalar(indices):
        indices = [indices]

    closest_manuscripts = [alive_manuscripts_in_region[i] for i in indices]

    # 2. Map manuscripts to witness instances
    candidate_instances = [manuscript_to_instance_map[ms.manuscript_id] for ms in closest_manuscripts]

    # 3. Reputation-based Ranking (with Preference for Antiquity)
    # Scribes prefer high reputation. If reputations are equal, they prefer
    # older manuscripts (lower birth_tick).
    def sort_key(inst_id: WitnessInstanceID) -> tuple[float, float]:
        if reputation_cache is not None and age_cache is not None:
            rep = reputation_cache[inst_id]
            birth_tick = age_cache[inst_id]
        else:
            node = graph.nodes[inst_id]
            rep = node["reputation"]
            birth_tick = node["birth_tick"]

        # We sort descending. Higher reputation comes first.
        # For equal reputation, smaller birth_tick (older) comes first.
        # So we use -birth_tick to make smaller values "larger" in descending sort.
        return (rep, -birth_tick)

    candidate_instances.sort(key=sort_key, reverse=True)

    # 4. Determine number of exemplars
    n = rng.choice([1, 2, 3], p=config.parent_num_distribution)
    n = min(n, len(candidate_instances))

    # 5. Final Selection
    return candidate_instances[:n]
