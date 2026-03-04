import pytest
from pydantic import ValidationError

from pasim.config.schema import SimulationConfig
from pasim.core.state import Region


def test_pa_config_valid():
    """Verify that a valid PA configuration is accepted."""
    config_data = {
        "total_ticks": 100,
        "text_length": 200,
        "p_region_migration": 0.05,
        "p_internal_relocation": 0.1,
        "reputation_distribution": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "demand_schedule": {0: 10},
        "pa_regime": "insertion",
        "pa_intervention_year": 50,
        "pa_intervention_region": "Asia Minor",
        "pa_innovator_reputation": 5.0,
    }
    config = SimulationConfig(**config_data)
    assert config.pa_regime == "insertion"
    assert config.pa_intervention_year == 50
    assert config.pa_intervention_region == Region.ASIA_MINOR
    assert config.pa_innovator_reputation == 5.0


def test_pa_config_invalid_regime():
    """Verify that an invalid PA regime is rejected."""
    config_data = {
        "total_ticks": 100,
        "text_length": 200,
        "reputation_distribution": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "demand_schedule": {0: 10},
        "pa_regime": "invalid_regime",
        "pa_intervention_year": 50,
        "pa_intervention_region": "Asia Minor",
        "pa_innovator_reputation": 5.0,
    }
    with pytest.raises(ValidationError) as excinfo:
        SimulationConfig(**config_data)
    assert "pa_regime" in str(excinfo.value)


def test_pa_config_intervention_year_out_of_bounds():
    """Verify that an intervention year exceeding total_ticks is rejected."""
    config_data = {
        "total_ticks": 100,
        "text_length": 200,
        "reputation_distribution": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "demand_schedule": {0: 10},
        "pa_regime": "insertion",
        "pa_intervention_year": 150,
        "pa_intervention_region": "Asia Minor",
        "pa_innovator_reputation": 5.0,
    }
    with pytest.raises(ValidationError) as excinfo:
        SimulationConfig(**config_data)
    assert "pa_intervention_year" in str(excinfo.value)


def test_pa_config_invalid_region():
    """Verify that an invalid region is rejected."""
    config_data = {
        "total_ticks": 100,
        "text_length": 200,
        "reputation_distribution": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "demand_schedule": {0: 10},
        "pa_regime": "insertion",
        "pa_intervention_year": 50,
        "pa_intervention_region": "Invalid Region",
        "pa_innovator_reputation": 5.0,
    }
    with pytest.raises(ValidationError) as excinfo:
        SimulationConfig(**config_data)
    assert "pa_intervention_region" in str(excinfo.value)


def test_pa_config_invalid_reputation():
    """Verify that an innovator reputation outside [1, 5] is rejected."""
    base_data = {
        "total_ticks": 100,
        "text_length": 200,
        "reputation_distribution": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "demand_schedule": {0: 10},
        "pa_regime": "insertion",
        "pa_intervention_year": 50,
        "pa_intervention_region": "Asia Minor",
    }

    # Test below 1.0
    with pytest.raises(ValidationError):
        SimulationConfig(**base_data, pa_innovator_reputation=0.9)

    # Test above 5.0
    with pytest.raises(ValidationError):
        SimulationConfig(**base_data, pa_innovator_reputation=5.1)


def test_pa_config_omission_regime():
    """Verify that 'omission' regime is also accepted."""
    config_data = {
        "total_ticks": 100,
        "text_length": 200,
        "reputation_distribution": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "demand_schedule": {0: 10},
        "pa_regime": "omission",
        "pa_intervention_year": 50,
        "pa_intervention_region": "Levant",
        "pa_innovator_reputation": 4.5,
    }
    config = SimulationConfig(**config_data)
    assert config.pa_regime == "omission"
