from typing import Dict

import numpy as np

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_snapshot import GenealogyNode, GenealogySnapshot
from pasim.core.rng import RNGContext
from pasim.core.scribal_rules import apply_scribal_rule
from pasim.core.text_initialisation import make_initial_text


class TextReplayEngine:
    """
    Replays textual transmission and mutation over a fixed genealogy graph.
    """

    def __init__(self, config: SimulationConfig, snapshot: GenealogySnapshot, seed: int):
        self.config = config
        self.snapshot = snapshot
        self.seed = seed
        self.rng_context = RNGContext(seed)
        self.instance_texts: Dict[str, np.ndarray] = {}

    def run(self) -> Dict[str, np.ndarray]:
        """
        Executes the replay, traversing nodes in birth order.
        """

        # Sort nodes by birth tick to ensure parents are processed before children.
        # Within the same tick, sort by the numeric value of instance_id (e.g., I10 > I2)
        # Assuming IDs are like "I1", "I2", ...
        def sort_key(node: GenealogyNode):
            # Extract number from I1, I2 etc.
            try:
                numeric_id = int(node.instance_id[1:])
            except (ValueError, TypeError):
                # Fallback to hash if it's not a standard ID, to keep it int
                numeric_id = hash(node.instance_id)
            return (node.birth_tick, numeric_id)

        sorted_nodes = sorted(self.snapshot.nodes, key=sort_key)

        # Each node needs its own RNG for scribal rules to ensure independence
        # if the graph structure is the same but texts differ.
        # We spawn one RNG per node from the master replay seed.
        rngs = self.rng_context.spawn(len(sorted_nodes))

        for node, rng in zip(sorted_nodes, rngs):
            self._process_node(node, rng)

        return self.instance_texts

    def _process_node(self, node: GenealogyNode, rng: np.random.Generator):
        """Processes a single node, generating its text."""
        if not node.parent_ids:
            # Root node / autograph
            # In Phase 2B, we don't have PA genome overrides yet,
            # but we know 0 is legal.
            text = make_initial_text(self.config)
            # TODO: In Phase 3, apply PA regime specific autograph here.
        else:
            parent_texts = [self.instance_texts[pid] for pid in node.parent_ids]
            text = apply_scribal_rule(exemplar_texts=parent_texts, rng=rng, reputation=node.reputation, config=self.config)

        # TODO: In Phase 3, apply PA intervention at config.pa_intervention_year
        # and config.pa_intervention_region if this node matches.

        self.instance_texts[node.instance_id] = text
