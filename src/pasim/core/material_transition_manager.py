"""
Provides a model for the historical transition of writing materials.

This component manages a time-dependent schedule that dictates the probability
distribution of materials for newly created manuscripts. It is a persistent rule,
not a discrete event, and only influences the `material` attribute of manuscripts
at the point of their creation (spawning).

The `MaterialTransitionManager` ensures that as the simulation progresses through
historical ticks, the types of materials predominantly used for new manuscripts
reflect historical technological shifts (e.g., from papyrus to parchment to paper).

Key aspects:
-   **Not a HistoricalEvent**: This is explicitly NOT a `HistoricalEvent`. It does not
    act on existing manuscripts or represent a "shock." Instead, it modifies the
    parameters of a continuous process (manuscript spawning).
-   **New Manuscripts Only**: Existing manuscripts retain their material. This
    model only affects the `material` assigned to new manuscripts spawned due
    to demand.
-   **Parameter-Driven**: The transition schedule is defined entirely by user
    parameters, allowing for flexible historical scenarios. Schedules are
    defined as a sequence of `start_tick` and associated material probability
    `distribution` dictionaries.

The manager loads a schedule and, for any given tick, determines the active
material distribution to sample from.
"""

from typing import Any, Dict, List

import numpy as np

from .state import Material


class MaterialTransitionManager:
    """
    Manages the time-dependent probability distribution for manuscript materials.
    """

    def __init__(self, schedule_configs: List[Dict[str, Any]]):
        """
        Initializes the manager with a schedule of material distributions.

        Args:
            schedule_configs: A list of dictionaries, each defining a
                              `start_tick` and a `distribution` for materials.
                              Example:
                              [
                                {'start_tick': 0, 'distribution': {'papyrus': 0.9, 'parchment': 0.1}},
                                {'start_tick': 300, 'distribution': {'papyrus': 0.2, 'parchment': 0.7}},
                              ]
        Raises:
            ValueError: If the schedule is invalid (e.g., non-positive probabilities,
                        probabilities not summing to 1, unknown materials).
        """
        if not schedule_configs:
            # If no schedule is provided, material assignment should fall back
            # to some default or be explicitly configured for a single choice.
            # For now, we raise an error to ensure configuration.
            raise ValueError("Material transition schedule cannot be empty. Provide at least one entry, e.g., for start_tick 0.")

        # Sort the schedule by start_tick to ensure correct lookup
        self._schedule = sorted(schedule_configs, key=lambda x: x["start_tick"])

        # Pre-process and validate distributions
        self._processed_schedule: List[Dict[str, Any]] = []
        for entry in self._schedule:
            start_tick = entry["start_tick"]
            distribution_dict = entry["distribution"]

            materials_list = []  # List of Material enum members
            probabilities_list = []  # List of floats
            total_prob = 0.0

            for mat_str, prob in distribution_dict.items():
                try:
                    material_enum = Material[mat_str.upper()]
                except KeyError:
                    raise ValueError(
                        f"Unknown material '{mat_str}' in schedule at tick {start_tick}. "
                        f"Valid materials are: {[m.name.lower() for m in Material]}."
                    )

                if prob < 0:
                    raise ValueError(f"Material probability for '{mat_str}' at tick {start_tick} is negative: {prob}.")

                materials_list.append(material_enum)
                probabilities_list.append(prob)
                total_prob += prob

            if not np.isclose(total_prob, 1.0):
                raise ValueError(f"Material probabilities at tick {start_tick} do not sum to 1.0. Got {total_prob}.")

            self._processed_schedule.append({
                "start_tick": start_tick,
                "materials": materials_list,
                "probabilities": probabilities_list,
            })

    def get_active_distribution(self, tick: int) -> Dict[str, Any]:
        """
        Returns the active material distribution for a given tick.

        Args:
            tick: The current simulation tick.

        Returns:
            Dict[str, Any]: A dictionary containing 'materials' and 'probabilities'.
        """
        active_distribution = None
        # Iterate backwards to find the most recent start_tick <= current tick
        for entry in reversed(self._processed_schedule):
            if tick >= entry["start_tick"]:
                active_distribution = entry
                break

        if active_distribution is None:
            # This case should be prevented by validation ensuring start_tick 0 exists
            raise RuntimeError(
                f"No active material distribution found for tick {tick}. "
                "Ensure a distribution with start_tick <= current_tick is always provided."
            )

        return active_distribution

    def get_material_for_tick(self, tick: int, rng: np.random.Generator) -> Material:
        """
        Samples a material based on the active distribution for the given tick.

        Args:
            tick: The current simulation tick.
            rng: The NumPy random number generator for deterministic sampling.

        Returns:
            Material: The sampled material.
        """
        active_distribution = self.get_active_distribution(tick)
        sampled_material = rng.choice(a=active_distribution["materials"], p=active_distribution["probabilities"])
        return sampled_material
