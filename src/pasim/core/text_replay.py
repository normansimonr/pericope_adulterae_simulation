from typing import Any, Dict, Optional

import numpy as np

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_snapshot import GenealogyNode, GenealogySnapshot
from pasim.core.rng import RNGContext
from pasim.core.scribal_rules import apply_scribal_rule
from pasim.core.text_initialisation import make_initial_text


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

        # Incremental Metric Tracking
        self.text_length = config.text_length
        self.autograph_text: Optional[np.ndarray] = None
        self.all_with_pa_count = 0
        self.total_processed_count = 0
        # Buffer for ideal majority: [segment_index][value_0_to_5]
        self.ideal_majority_counts = np.zeros((self.text_length, 6), dtype=np.int32)

    def _select_innovator(self) -> Optional[str]:
        """
        Finds the innovator node for the current PA regime by looking for the
        pre-assigned tag in the genealogy snapshot.
        """
        current_regime = self.config.pa_regime
        for node in self.snapshot.nodes:
            if current_regime in node.pa_intervention_regimes:
                return node.instance_id

        # If not found, it's a critical error.
        raise RuntimeError(
            f"No eligible nodes found for PA intervention at year {self.config.pa_intervention_year} "
            f"in region {self.config.pa_intervention_region.value}"
        )

    def run(self, survivor_ids: Optional[set[str]] = None, return_all_texts: bool = True) -> Dict[str, np.ndarray]:
        """
        Executes the replay, traversing nodes in birth order.
        If return_all_texts is False, only the survivor texts are kept in memory,
        and metrics are computed incrementally.
        """
        sorted_nodes = self._get_sorted_nodes()
        rngs = self.rng_context.spawn(len(sorted_nodes))

        # Track child counts for memory-efficient cleanup
        child_counts = self._get_child_counts()

        for node, rng in zip(sorted_nodes, rngs):
            text = self._process_node(node, rng)

            # 1. Update Incremental Metrics
            is_autograph = not node.parent_ids
            if is_autograph:
                self.autograph_text = text.copy()
            else:
                # Add to ideal majority tally (exclude autograph)
                self.ideal_majority_counts[np.arange(self.text_length), text] += 1
                self.total_processed_count += 1
                if np.any(text != 0):
                    self.all_with_pa_count += 1

            # 2. Memory Management: Store or discard text
            self.instance_texts[node.instance_id] = text

            if not return_all_texts:
                self._cleanup_unused_texts(node, child_counts, survivor_ids)

        if not self.intervention_applied:
            raise RuntimeError(f"Intervention not applied. Regime: {self.config.pa_regime}")

        return self.instance_texts

    def get_ideal_metrics(self) -> Dict[str, Any]:
        """Returns metrics computed incrementally during the run."""
        # For each segment, the reading with the highest count is the majority
        ideal_majority = np.argmax(self.ideal_majority_counts, axis=1).astype(np.int16)

        pct_all_with_pa = 0.0
        if self.total_processed_count > 0:
            pct_all_with_pa = self.all_with_pa_count / self.total_processed_count

        pct_ideal_majority_disagree = 0.0
        if self.autograph_text is not None:
            disagree_count = np.sum(ideal_majority != self.autograph_text)
            pct_ideal_majority_disagree = disagree_count / self.text_length

        return {
            "ideal_majority_text_segments": ideal_majority.tolist(),
            "pct_all_witnesses_with_pa": pct_all_with_pa,
            "pct_ideal_majority_disagree_autograph": pct_ideal_majority_disagree,
            "autograph_text": self.autograph_text,
        }

    def _get_sorted_nodes(self):
        def sort_key(node: GenealogyNode):
            try:
                numeric_id = int(node.instance_id[1:])
            except (ValueError, TypeError):
                numeric_id = hash(node.instance_id)
            return (node.birth_tick, numeric_id)

        return sorted(self.snapshot.nodes, key=sort_key)

    def _get_child_counts(self):
        counts = {node.instance_id: 0 for node in self.snapshot.nodes}
        for node in self.snapshot.nodes:
            for pid in node.parent_ids:
                if pid in counts:
                    counts[pid] += 1
        return counts

    def _cleanup_unused_texts(self, current_node, child_counts, survivor_ids):
        """Discards parent texts that are no longer needed for copying or sampling."""
        for pid in current_node.parent_ids:
            if pid in child_counts:
                child_counts[pid] -= 1
                if child_counts[pid] == 0:
                    # If it's not a survivor, we don't need it anymore
                    if survivor_ids is None or pid not in survivor_ids:
                        if pid in self.instance_texts:
                            del self.instance_texts[pid]

    def _process_node(self, node: GenealogyNode, rng: np.random.Generator):
        """Processes a single node, generating its text with potential PA override."""
        is_innovator = node.instance_id == self.innovator_id

        if not node.parent_ids:
            text = make_initial_text(self.config)
        else:
            parent_texts = [self.instance_texts[pid] for pid in node.parent_ids]
            reputation = node.reputation
            if is_innovator:
                reputation = int(self.config.pa_innovator_reputation)
            text = apply_scribal_rule(exemplar_texts=parent_texts, rng=rng, reputation=reputation, config=self.config)

        if is_innovator:
            length = self.config.text_length
            if self.config.pa_regime == "insertion":
                text = np.ones(length, dtype=np.int16)
            elif self.config.pa_regime == "omission":
                text = np.zeros(length, dtype=np.int16)
            self.intervention_applied = True

        return text
