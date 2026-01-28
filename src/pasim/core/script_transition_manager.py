"""
Provides a model for the historical transition of script styles.

This component manages a time-dependent schedule that dictates the probability
distribution of scripts for newly created witnesses. It is a persistent rule,
not a discrete event, and only influences the `script` attribute of witnesses
at the point of their creation.

The `ScriptTransitionManager` ensures that as the simulation progresses through
historical ticks, the types of scripts predominantly used for new witnesses
reflect historical shifts (e.g., from uncial to minuscule).

Key aspects:
-   **Not a HistoricalEvent**: This is explicitly NOT a `HistoricalEvent`. It does not
    act on existing witnesses or represent a "shock." Instead, it modifies the
    parameters of a continuous process (witness creation).
-   **New Witnesses Only**: Existing witnesses are not affected. This
    model only affects the `script` assigned to new witnesses.
-   **Parameter-Driven**: The transition schedule is defined entirely by user
    parameters, allowing for flexible historical scenarios.

The manager loads a schedule and, for any given tick, determines the active
script distribution to sample from.
"""

from typing import Dict, Any, List
import numpy as np

from .state import Script


class ScriptTransitionManager:
    """
    Manages the time-dependent probability distribution for witness scripts.
    """

    def __init__(self, schedule_configs: List[Dict[str, Any]]):
        """
        Initializes the manager with a schedule of script distributions.

        Args:
            schedule_configs: A list of dictionaries, each defining a
                              `start_tick` and a `distribution` for scripts.
                              Example:
                              [
                                {'start_tick': 0, 'distribution': {'uncial': 1.0}},
                                {'start_tick': 350, 'distribution': {'uncial': 0.6, 'minuscule': 0.4}},
                              ]
        Raises:
            ValueError: If the schedule is invalid.
        """
        if not schedule_configs:
            raise ValueError("Script transition schedule cannot be empty.")

        self._schedule = sorted(schedule_configs, key=lambda x: x['start_tick'])

        self._processed_schedule: List[Dict[str, Any]] = []
        for entry in self._schedule:
            start_tick = entry['start_tick']
            distribution_dict = entry['distribution']

            scripts_list = []
            probabilities_list = []
            total_prob = 0.0

            for script_str, prob in distribution_dict.items():
                try:
                    script_enum = Script[script_str.upper()]
                except KeyError:
                    raise ValueError(f"Unknown script '{script_str}' in schedule at tick {start_tick}. "
                                     f"Valid scripts are: {[s.name.lower() for s in Script]}.")

                if prob < 0:
                    raise ValueError(f"Script probability for '{script_str}' at tick {start_tick} is negative: {prob}.")

                scripts_list.append(script_enum)
                probabilities_list.append(prob)
                total_prob += prob

            if not np.isclose(total_prob, 1.0):
                raise ValueError(f"Script probabilities at tick {start_tick} do not sum to 1.0. Got {total_prob}.")

            self._processed_schedule.append({
                'start_tick': start_tick,
                'scripts': scripts_list,
                'probabilities': probabilities_list
            })

    def get_script_for_tick(self, tick: int, rng: np.random.Generator) -> Script:
        """
        Samples a script based on the active distribution for the given tick.

        Args:
            tick: The current simulation tick.
            rng: The NumPy random number generator for deterministic sampling.

        Returns:
            Script: The sampled script.
        """
        active_distribution = None
        for entry in reversed(self._processed_schedule):
            if tick >= entry['start_tick']:
                active_distribution = entry
                break

        if active_distribution is None:
            raise RuntimeError(f"No active script distribution found for tick {tick}.")

        sampled_script = rng.choice(
            a=active_distribution['scripts'],
            p=active_distribution['probabilities']
        )
        return sampled_script
