from typing import Dict, Optional

import numpy as np
from scipy.spatial import KDTree

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_snapshot import GenealogyNode, GenealogySnapshot
from pasim.core.rng import RNGContext
from pasim.core.scribal_rules import apply_scribal_rule
from pasim.core.text_initialisation import make_initial_text

# Default radius for innovator selection neighbor count if not found elsewhere.
# This represents 10% of the standard 100x100 region size.
PA_INTERVENTION_RADIUS = 10.0


class TextReplayEngine:
    """
    Replays textual transmission and mutation over a fixed genealogy graph.
    Includes support for regime-dependent PA intervention.
    """

    def __init__(self, config: SimulationConfig, snapshot: GenealogySnapshot, seed: int):
        self.config = config
        self.snapshot = snapshot
        self.seed = seed
        self.rng_context = RNGContext(seed)
        self.instance_texts: Dict[str, np.ndarray] = {}
        self.intervention_applied = False
        self.innovator_id: Optional[str] = self._select_innovator()

    def _select_innovator(self) -> str:
        """
        Deterministically selects the innovator node for PA intervention.
        The innovator is the node at the intervention year and region with
        the highest neighbor count.
        """
        eligible_nodes = [
            n
            for n in self.snapshot.nodes
            if n.birth_tick == self.config.pa_intervention_year and n.region == self.config.pa_intervention_region
        ]

        if not eligible_nodes:
            raise RuntimeError(
                f"No eligible nodes found for PA intervention at year {self.config.pa_intervention_year} "
                f"in region {self.config.pa_intervention_region.value}"
            )

        # Reference pool: all nodes alive at the intervention year in the same region.
        # A node is alive if birth_tick <= year < death_tick.
        reference_nodes = [
            n
            for n in self.snapshot.nodes
            if n.birth_tick <= self.config.pa_intervention_year < n.death_tick and n.region == self.config.pa_intervention_region
        ]

        # If reference_nodes is empty (should not happen if eligible_nodes exists),
        # fallback to eligible_nodes itself.
        if not reference_nodes:
            reference_nodes = eligible_nodes

        ref_locations = np.array([n.location for n in reference_nodes])
        tree = KDTree(ref_locations)

        best_node_id = None
        max_count = -1

        # Helper to get numeric part of ID for deterministic tie-breaking
        def get_numeric_id(node_id: str) -> int:
            try:
                return int(node_id[1:])
            except (ValueError, TypeError):
                return hash(node_id)

        for node in eligible_nodes:
            # Neighbor count within fixed radius
            count = len(tree.query_ball_point(node.location, r=PA_INTERVENTION_RADIUS))

            node_numeric_id = get_numeric_id(node.instance_id)

            if count > max_count:
                max_count = count
                best_node_id = node.instance_id
            elif count == max_count:
                # Tie-break: lowest numeric instance_id
                if best_node_id is None or node_numeric_id < get_numeric_id(best_node_id):
                    best_node_id = node.instance_id

        if best_node_id is None:
            raise RuntimeError("Failed to select an innovator node.")

        return best_node_id

    def run(self) -> Dict[str, np.ndarray]:
        """
        Executes the replay, traversing nodes in birth order.
        """

        # Sort nodes by birth tick to ensure parents are processed before children.
        # Within the same tick, sort by the numeric value of instance_id (e.g., I10 > I2)
        def sort_key(node: GenealogyNode):
            try:
                numeric_id = int(node.instance_id[1:])
            except (ValueError, TypeError):
                numeric_id = hash(node.instance_id)
            return (node.birth_tick, numeric_id)

        sorted_nodes = sorted(self.snapshot.nodes, key=sort_key)

        # Each node needs its own RNG for scribal rules.
        rngs = self.rng_context.spawn(len(sorted_nodes))

        for node, rng in zip(sorted_nodes, rngs):
            self._process_node(node, rng)

        # Guarantee single execution: check if intervention was applied
        if not self.intervention_applied:
            raise RuntimeError(
                f"Intervention was not applied despite replay completing. "
                f"Target year: {self.config.pa_intervention_year}, target ID: {self.innovator_id}"
            )

        return self.instance_texts

    def _process_node(self, node: GenealogyNode, rng: np.random.Generator):
        """Processes a single node, generating its text with potential PA override."""
        is_innovator = node.instance_id == self.innovator_id

        # Normal textual transmission
        if not node.parent_ids:
            # Root node / autograph
            text = make_initial_text(self.config)
        else:
            parent_texts = [self.instance_texts[pid] for pid in node.parent_ids]

            # Use innovator reputation if this is the innovator
            reputation = node.reputation
            if is_innovator:
                reputation = int(self.config.pa_innovator_reputation)

            text = apply_scribal_rule(exemplar_texts=parent_texts, rng=rng, reputation=reputation, config=self.config)

        # Apply PA Intervention override at birth
        if is_innovator:
            length = self.config.text_length
            if self.config.pa_regime == "insertion":
                # Autograph was all 0, innovator introduces 1s
                text = np.ones(length, dtype=np.int16)
            elif self.config.pa_regime == "omission":
                # Autograph was all 1, innovator introduces 0s
                text = np.zeros(length, dtype=np.int16)

            self.intervention_applied = True

        self.instance_texts[node.instance_id] = text
