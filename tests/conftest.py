"""
This file contains shared fixtures and configuration for the pytest framework.
"""

from typing import Generator, List

import pytest

from pasim.analysis.text_view import print_textual_diffs
from pasim.core.simulation_state import GenerationState


def pytest_addoption(parser):
    """Adds the --show-text-diffs command-line option to pytest."""
    parser.addoption(
        "--show-text-diffs",
        action="store_true",
        default=False,
        help="Show textual diffs after simulation tests",
    )


@pytest.fixture(autouse=True)
def state_collector_fixture(request) -> Generator[List[GenerationState], None, None]:
    """
    An autouse fixture that collects GenerationState objects during test execution
    and visualizes them in its teardown phase if --show-text-diffs is enabled.
    """
    collected_states: List[GenerationState] = []
    yield collected_states  # Yield the list for tests to append states to

    # This part runs after the test function has completed
    if request.config.getoption("--show-text-diffs"):
        for state in collected_states:
            print_textual_diffs(state)
