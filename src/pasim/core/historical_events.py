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

from typing import Optional, List, Set, Any
from dataclasses import dataclass, field
import numpy as np

from .state import GenerationState


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

    def apply(self, state: GenerationState, rng: np.random.Generator, params: Any) -> None:
        """
        Applies the historical event's effect to the simulation state.

        This method is intended to be overridden by concrete event implementations.

        Args:
            state: The current simulation state.
            rng: The simulation's random number generator.
            params: The global simulation parameters.

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


class HistoricalEventManager:
    """
    Manages and applies all historical events during a simulation run.

    This class is responsible for dispatching events at the correct time and
    in a deterministic order. It does not contain any event-specific logic
    itself, but rather orchestrates the execution of `HistoricalEvent` objects.
    """

    def __init__(self, events: Optional[List[HistoricalEvent]] = None):
        """
        Initializes the manager with a list of historical events.

        Args:
            events: A list of HistoricalEvent objects to manage.
        """
        # Sort events deterministically to ensure stable application order.
        # Sort first by start_tick, then by the event's class name.
        self._events = sorted(
            events or [],
            key=lambda e: (e.start_tick, e.__class__.__name__)
        )

    def apply_events_for_tick(self, state: GenerationState, rng: np.random.Generator, params: Any) -> None:
        """
        Finds and applies all active historical events for the current tick.

        This method reads the current tick from the state and iterates through
        its list of events, applying any that are active.

        Args:
            state: The current simulation state, which includes the current tick.
            rng: The simulation's random number generator.
            params: The global simulation parameters.
        """
        current_tick = state.tick
        active_events = [event for event in self._events if event.is_active(current_tick)]

        for event in active_events:
            # The apply method of the concrete event will handle region filtering.
            event.apply(state, rng, params)
