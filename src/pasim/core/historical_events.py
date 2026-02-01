"""
Provides the infrastructure for modeling exogenous historical events.

This module introduces a clear separation between two types of simulation dynamics:
1.  **Mechanistic Rules**: These are the fundamental, continuous processes of
    the simulation, such as manuscript death, copying (demand-based spawning),
    and migration. They are local, emergent, and happen tick by tick according
    to a fixed set of rules.
2.  **Historical Rules**: These are discrete, time-bound, and often global or
    regional "shocks" or transitions that affect the simulation state from the
    outside. Examples include imperial persecutions, the transition from papyrus
    to parchment, or changes in dominant scriptoria.

The `HistoricalEventManager` provides a deterministic, plug-in architecture for
these historical rules. It ensures that events are applied at the correct time
and in a stable order, without altering the core batch execution logic. This
design allows researchers to easily add, remove, or modify historical scenarios
without changing the underlying simulation engine.

Each `HistoricalEvent` is a self-contained unit of logic that is triggered by
the manager at the appropriate tick.
"""

from typing import Optional, List, Set, Any, Dict
from dataclasses import dataclass, field
import numpy as np
import math

from .simulation_state import GenerationState


@dataclass
class HistoricalEvent:
    """
    Represents a single, exogenous historical event or long-term transition.

    This is an abstract base structure. Concrete historical rules (like a
    persecution event or a material transition) will be implemented as
    subclasses or wrappers of this structure.

    Attributes:
        start_tick: The simulation tick at which the event begins.
        end_tick: If not None, the simulation tick at which the event ends.
                  If None, the event is instantaneous and occurs only at
                  start_tick.
        regions: If not None, a set of region identifiers to which this event
                 is restricted. If None, the event is global.
    """
    start_tick: int
    end_tick: Optional[int] = None
    regions: Optional[Set[str]] = None

    def apply(self, state: GenerationState, rng: np.random.Generator) -> None:
        """
        Applies the historical event's effect to the simulation state.

        This method is intended to be overridden by concrete event implementations.

        Args:
            state: The current simulation state.
            rng: The simulation's random number generator.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Each historical event must implement the 'apply' method.")

    def is_active(self, tick: int) -> bool:
        """
        Determines if the event is active at a given tick.

        Args:
            tick: The current simulation tick.

        Returns:
            True if the event is active, False otherwise.
        """
        if self.end_tick is not None:
            return self.start_tick <= tick <= self.end_tick
        return tick == self.start_tick


@dataclass(kw_only=True)
class PersecutionEvent(HistoricalEvent):
    """
    A historical event that models persecution, destroying a fraction of manuscripts.

    This event simulates a historical shock where a proportion of manuscripts
    in specific regions are suddenly removed from the "alive" pool, making them
    unavailable for future copying. This is distinct from a manuscript's
    natural lifecycle (scheduled death) and represents an external, destructive force.

    Attributes:
        kill_proportion (float): The fraction of eligible manuscripts to be
                                 destroyed, in the interval [0.0, 1.0].
    """
    kill_proportion: float

    def __post_init__(self):
        if not (0.0 <= self.kill_proportion <= 1.0):
            raise ValueError(
                "kill_proportion must be between 0.0 and 1.0, "
                f"but got {self.kill_proportion}"
            )

    def apply(self, state: GenerationState, rng: np.random.Generator) -> None:
        """
        Applies the persecution effect.

        Identifies all eligible manuscripts, calculates the number to destroy
        based on `kill_proportion`, and randomly removes them from the set of
        `alive_manuscripts`.

        Args:
            state: The current simulation state.
            rng: The simulation's random number generator for deterministic selection.
        """
        if self.kill_proportion == 0.0:
            return

        # 1. Identify eligible manuscripts
        eligible_manuscript_ids = []
        for ms_id in state.alive_manuscripts:
            ms_obj = state.registries.manuscripts.get(ms_id)
            if self.regions is None or ms_obj.region.value in self.regions:
                eligible_manuscript_ids.append(ms_id)

        if not eligible_manuscript_ids:
            return

        # 2. Determine number to destroy
        n_to_destroy = math.floor(self.kill_proportion * len(eligible_manuscript_ids))
        if n_to_destroy == 0:
            return

        # 3. Randomly choose victims (IDs)
        eligible_array = np.array(list(eligible_manuscript_ids), dtype=object)
        victims_ids = rng.choice(
            eligible_array,
            size=n_to_destroy,
            replace=False
        )

        # 4. Kill them (remove from alive set)
        state.alive_manuscripts.difference_update(victims_ids)


def create_event_from_config(config: Dict[str, Any]) -> HistoricalEvent:
    """
    Factory function to create a HistoricalEvent from a configuration dictionary.
    """
    event_type = config.pop("event_type")
    
    if event_type == "persecution":
        # Ensure 'regions' is a set if it exists
        if 'regions' in config and config['regions'] is not None:
            config['regions'] = set(config['regions'])
        return PersecutionEvent(**config)
    
    raise ValueError(f"Unknown historical event type: {event_type}")


class HistoricalEventManager:
    """
    Manages and applies all historical events during a simulation run.
    """

    def __init__(self, event_configs: Optional[List[Dict[str, Any]]] = None):
        """
        Initializes the manager by building event objects from configuration dicts.

        Args:
            event_configs: A list of configuration dictionaries, each defining
                           a historical event.
        """
        events = [create_event_from_config(cfg.copy()) for cfg in event_configs or []]
        
        self._events = sorted(
            events,
            key=lambda e: (e.start_tick, e.__class__.__name__)
        )

    def apply_events_for_tick(self, state: GenerationState, rng: np.random.Generator) -> None:
        """
        Finds and applies all active historical events for the current tick.
        """
        current_tick = state.tick
        active_events = [event for event in self._events if event.is_active(current_tick)]

        for event in active_events:
            event.apply(state, rng)
