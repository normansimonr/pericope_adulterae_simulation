"""
This module provides logging utilities for the pasim project, including
decorators for performance monitoring and structured logging.
"""

import functools
import logging
import time
from typing import Any, Callable

from pasim.core.simulation_state import GenerationState


def log_tick_performance(logger: logging.Logger, log_every_n_ticks: int = 1) -> Callable[..., Any]:
    """
    A decorator that logs the performance and state of a simulation tick.

    This decorator is designed to wrap the `advance_tick` function. It logs
    key metrics before and after the tick execution, including the duration
    of the tick and the number of alive/total manuscripts.

    Args:
        logger: The logger instance to use for logging.
        log_every_n_ticks: Log only when the current tick is a multiple of this value.
                           Defaults to 1, logging every tick.

    Returns:
        A wrapper function.
    """

    def decorator(func: Callable[..., GenerationState]) -> Callable[..., GenerationState]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> GenerationState:
            state_before = kwargs.get("state") or (args[0] if args else None)
            if not state_before or not isinstance(state_before, GenerationState):
                # Cannot log if state is not provided
                return func(*args, **kwargs)

            tick = state_before.tick + 1
            log_level = logging.INFO

            if tick % log_every_n_ticks == 0:
                logger.log(log_level, f"Tick {tick:04d}: Starting... (Alive Manuscripts: {len(state_before.alive_manuscripts)})")

            start_time = time.perf_counter()
            state_after = func(*args, **kwargs)
            end_time = time.perf_counter()

            duration_ms = (end_time - start_time) * 1000

            if tick % log_every_n_ticks == 0:
                logger.log(
                    log_level,
                    f"Tick {tick:04d}: Completed in {duration_ms:.2f}ms. "
                    f"(Alive Manuscripts: {len(state_after.alive_manuscripts)}, "
                    f"Total Manuscripts: {len(state_after.registries.manuscripts)})",
                )

            return state_after

        return wrapper

    return decorator
