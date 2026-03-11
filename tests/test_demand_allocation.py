from math import ceil
from typing import Any, Dict

import pytest

from pasim.config.schema import DemandScheduleConfig, SimulationConfig
from pasim.core.genealogy_generator import _allocate_demand
from pasim.core.state import Region

# Re-define the expected distributions locally for testing since they were removed from the core
EXPECTED_DISTRIBUTIONS = {
    "0-2": {
        Region.ASIA_MINOR: 0.70,
        Region.LEVANT: 0.25,
        Region.EGYPT: 0.05,
    },
    "3-5": {
        Region.ASIA_MINOR: 0.55,
        Region.LEVANT: 0.25,
        Region.EGYPT: 0.20,
    },
    "6+": {
        Region.ASIA_MINOR: 1.00,
        Region.LEVANT: 0.00,
        Region.EGYPT: 0.00,
    },
}


@pytest.fixture
def default_config():
    return SimulationConfig(
        total_ticks=1000,
        reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        pa_regime="insertion",
        pa_intervention_year=300,
        pa_intervention_region="Asia Minor",
        pa_innovator_reputation=5.0,
        demand_schedule={0: 10},
    )


# Helper to validate that old region-based config raises an error
def assert_old_regional_demand_raises_error(demand_data: Dict[int, Dict[str, Any]]):
    with pytest.raises(ValueError) as excinfo:
        DemandScheduleConfig.model_validate(demand_data)
    assert "Demand value for tick" in str(excinfo.value) or "Input should be a valid integer" in str(excinfo.value)


# Test cases for _allocate_demand
def test_allocate_demand_century_0(default_config, aggregate_demand=10):
    """Test allocation for Century 0 (ticks 0-99)."""
    tick = 50
    expected_distribution = EXPECTED_DISTRIBUTIONS["0-2"]

    expected_regional_demand = {
        Region.ASIA_MINOR: ceil(aggregate_demand * expected_distribution[Region.ASIA_MINOR]),
        Region.LEVANT: ceil(aggregate_demand * expected_distribution[Region.LEVANT]),
        Region.EGYPT: ceil(aggregate_demand * expected_distribution[Region.EGYPT]),
    }

    actual_regional_demand = _allocate_demand(tick, aggregate_demand, default_config)
    assert actual_regional_demand == expected_regional_demand


def test_allocate_demand_century_3(default_config, aggregate_demand=20):
    """Test allocation for Century 3 (ticks 300-399)."""
    tick = 350
    expected_distribution = EXPECTED_DISTRIBUTIONS["3-5"]

    expected_regional_demand = {
        Region.ASIA_MINOR: ceil(aggregate_demand * expected_distribution[Region.ASIA_MINOR]),
        Region.LEVANT: ceil(aggregate_demand * expected_distribution[Region.LEVANT]),
        Region.EGYPT: ceil(aggregate_demand * expected_distribution[Region.EGYPT]),
    }

    actual_regional_demand = _allocate_demand(tick, aggregate_demand, default_config)
    assert actual_regional_demand == expected_regional_demand


def test_allocate_demand_century_6_onwards(default_config, aggregate_demand=5):
    """Test allocation for Century 6 onwards (>= 600 years)."""
    tick = 700
    expected_distribution = EXPECTED_DISTRIBUTIONS["6+"]

    expected_regional_demand = {
        Region.ASIA_MINOR: ceil(aggregate_demand * expected_distribution[Region.ASIA_MINOR]),
        Region.LEVANT: ceil(aggregate_demand * expected_distribution[Region.LEVANT]),
        Region.EGYPT: ceil(aggregate_demand * expected_distribution[Region.EGYPT]),
    }

    actual_regional_demand = _allocate_demand(tick, aggregate_demand, default_config)
    assert actual_regional_demand == expected_regional_demand


@pytest.mark.parametrize(
    "aggregate_demand, expected_total, expected_am, expected_lev, expected_eg",
    [
        (1, 3, 1, 1, 1),  # Century 0-2 distribution: 0.7, 0.25, 0.05. Ceil(1*0.7)=1, Ceil(1*0.25)=1, Ceil(1*0.05)=1
        (2, 4, 2, 1, 1),  # Ceil(2*0.7)=2, Ceil(2*0.25)=1, Ceil(2*0.05)=1. Corrected total from 3 to 4.
        (10, 11, 7, 3, 1),  # Ceil(10*0.7)=7, Ceil(10*0.25)=3, Ceil(10*0.05)=1. Corrected total from 13 to 11.
    ],
)
def test_allocate_demand_ceiling_and_over_allocation(
    default_config, aggregate_demand, expected_total, expected_am, expected_lev, expected_eg
):
    """
    Test ceiling rounding and ensure over-allocation is allowed for Century 0-2.
    Expected total is sum of individual ceil-rounded regional demands.
    """
    tick = 50  # Century 0
    actual_regional_demand = _allocate_demand(tick, aggregate_demand, default_config)

    assert actual_regional_demand[Region.ASIA_MINOR] == expected_am
    assert actual_regional_demand[Region.LEVANT] == expected_lev
    assert actual_regional_demand[Region.EGYPT] == expected_eg
    assert sum(actual_regional_demand.values()) == expected_total
    assert sum(actual_regional_demand.values()) >= aggregate_demand


def test_allocate_demand_zero_aggregate(default_config):
    """Test allocation when aggregate demand is zero."""
    tick = 150  # Century 1
    aggregate_demand = 0
    actual_regional_demand = _allocate_demand(tick, aggregate_demand, default_config)
    assert actual_regional_demand == {region: 0 for region in Region}


def test_allocate_demand_deterministic(default_config):
    """Verify that allocation is deterministic for same tick and aggregate."""
    tick = 123
    aggregate_demand = 17

    result1 = _allocate_demand(tick, aggregate_demand, default_config)
    result2 = _allocate_demand(tick, aggregate_demand, default_config)

    assert result1 == result2


# Test that old region-based config raises an error
# This test relies on the DemandScheduleConfig schema failing validation
def test_old_regional_config_raises_error():
    """
    Test that the old region-based demand schedule format now raises a validation error.
    """
    old_demand_format = {0: {"Asia Minor": 1, "Egypt": 1}, 5: {"Asia Minor": 2, "Egypt": 1}}

    # We expect a ValueError due to the type mismatch (Dict[Region, int] vs int)
    with pytest.raises(Exception) as excinfo:  # Broad exception because Pydantic can wrap it
        # Attempt to validate with the old format
        # Pydantic v2 will expect an int where it finds a dict, leading to a ValidationError
        DemandScheduleConfig.model_validate(old_demand_format)

    # Check for specific error message related to expected type
    assert "Input should be a valid integer" in str(excinfo.value)
