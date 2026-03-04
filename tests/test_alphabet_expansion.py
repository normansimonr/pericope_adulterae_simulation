import numpy as np
import pytest

from pasim.config.schema import SimulationConfig
from pasim.core.mutation import mutate_tagged_string
from pasim.core.tagged_string_constraints import (
    LEGAL_SEGMENT_VALUES,
    is_valid_segment_value,
    sample_alternative_value,
    validate_tagged_string,
)
from pasim.core.text_initialisation import make_initial_text


@pytest.fixture
def config():
    return SimulationConfig(
        total_ticks=10, text_length=10, demand_schedule={0: 1}, reputation_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
    )


def test_zero_is_legal():
    assert 0 in LEGAL_SEGMENT_VALUES
    assert is_valid_segment_value(0)


def test_initial_text_is_all_zeros(config):
    text = make_initial_text(config)
    assert np.all(text == 0)
    assert text.dtype == np.int16
    # Should pass validation
    validate_tagged_string(text, config)


def test_mutation_can_produce_zero(config):
    rng = np.random.default_rng(42)
    # Use a larger text to avoid random failure (1/6 chance per segment)
    large_config = SimulationConfig(
        total_ticks=1, text_length=100, demand_schedule={0: 1}, reputation_distribution={1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    )
    # Start with all 1s
    text = np.ones(large_config.text_length, dtype=np.int16)
    # Mutate all segments
    mutated = mutate_tagged_string(text, rng, expected_proportion=1.0, config=large_config)

    # Check if 0 is present in mutated text
    assert 0 in mutated
    validate_tagged_string(mutated, large_config)


def test_mutation_can_remove_zero(config):
    rng = np.random.default_rng(42)
    # Use a larger text
    large_config = SimulationConfig(
        total_ticks=1, text_length=100, demand_schedule={0: 1}, reputation_distribution={1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    )
    # Start with all 0s
    text = np.zeros(large_config.text_length, dtype=np.int16)
    # Mutate all segments
    mutated = mutate_tagged_string(text, rng, expected_proportion=1.0, config=large_config)

    # All segments should be different from 0
    assert 0 not in mutated
    validate_tagged_string(mutated, large_config)


def test_mutation_determinism(config):
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    text = make_initial_text(config)

    mutated1 = mutate_tagged_string(text, rng1, expected_proportion=0.5, config=config)
    mutated2 = mutate_tagged_string(text, rng2, expected_proportion=0.5, config=config)

    np.testing.assert_array_equal(mutated1, mutated2)


def test_sample_alternative_value_includes_zero():
    rng = np.random.default_rng(789)
    # If current is 1, can we get 0?
    results = set()
    for _ in range(100):
        results.add(sample_alternative_value(np.int16(1), rng))
    assert 0 in results
    assert 1 not in results
    assert results.issubset(set(LEGAL_SEGMENT_VALUES))


def test_sample_alternative_value_from_zero():
    rng = np.random.default_rng(789)
    # If current is 0, can we get others?
    results = set()
    for _ in range(100):
        results.add(sample_alternative_value(np.int16(0), rng))
    assert 0 not in results
    assert results.issubset({1, 2, 3, 4, 5})
