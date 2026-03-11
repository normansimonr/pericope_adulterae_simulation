import numpy as np
import pytest

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_generator import run_genealogy_generator
from pasim.core.lifespan import sample_lifespan
from pasim.core.rng import RNGContext
from pasim.core.state import Material, Region


# Helper to create a dummy config for run_genealogy_generator
def get_dummy_config(total_ticks=10, text_length=10, reputation_distribution=None, **kwargs):
    if reputation_distribution is None:
        reputation_distribution = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
    config_dict = {
        "total_ticks": total_ticks,
        "text_length": text_length,
        "p_region_migration": 0.0,
        "p_internal_relocation": 0.0,
        "reputation_distribution": reputation_distribution,
        "persecutions": [],
        "material_transitions": [{"start_tick": 0, "distribution": {"papyrus": 1.0}}],
        "script_transitions": [{"start_tick": 0, "distribution": {"uncial": 1.0}}],
        "demand_schedule": {0: 10},
        "pa_regime": "insertion",
        "pa_intervention_year": 0,
        "pa_intervention_region": "Asia Minor",
        "pa_innovator_reputation": 5.0,
        **kwargs,
    }
    return SimulationConfig(**config_dict)


@pytest.fixture
def rng():
    """Fixture for a reproducible RNG."""
    return RNGContext(seed=42).spawn(1)[0]


@pytest.fixture
def config():
    """Fixture for a default SimulationConfig."""
    return get_dummy_config()


def test_lifespan_min_one_year(rng: np.random.Generator, config: SimulationConfig):
    """Verify that sampled lifespan is always at least 1."""
    material = Material.PAPYRUS
    region = Region.EGYPT
    lifespans = [sample_lifespan(material, region, rng, config) for _ in range(1000)]
    assert all(ls >= 1 for ls in lifespans)


def test_lifespan_unknown_material_raises_error(rng: np.random.Generator, config: SimulationConfig):
    """Verify that an unknown material (as a string value) raises a ValueError."""
    # We need to bypass the Material enum validation to test the internal logic
    # but Material is an Enum, so Material("unknown") will raise ValueError anyway.
    # The actual check in _get_lognormal_params is for material.value not in lifespan_params.
    # If we pass a valid Material enum but it's not in the config, it should raise.

    # Create a config with missing material
    limited_config = get_dummy_config(lifespan_parameters={"papyrus": {"Egypt": {"mu": 5.0, "sigma": 0.5}}})

    with pytest.raises(ValueError, match="Unsupported material for lifespan calculation: parchment"):
        sample_lifespan(Material.PARCHMENT, Region.EGYPT, rng, limited_config)


def test_lifespan_unknown_region_raises_error(rng: np.random.Generator, config: SimulationConfig):
    """Verify that an unknown region raises a ValueError."""
    # Create a config with missing region for a material
    limited_config = get_dummy_config(lifespan_parameters={"papyrus": {"Egypt": {"mu": 5.0, "sigma": 0.5}}})

    with pytest.raises(ValueError, match="Unsupported region for lifespan calculation: Asia Minor with material papyrus"):
        sample_lifespan(Material.PAPYRUS, Region.ASIA_MINOR, rng, limited_config)


def test_lifespan_determinism(rng: np.random.Generator, config: SimulationConfig):
    """Verify that lifespan sampling is deterministic for a fixed seed."""
    material = Material.PARCHMENT
    region = Region.ASIA_MINOR

    # Generate two sets of lifespans with the same RNG state
    rng1 = RNGContext(seed=123).spawn(1)[0]
    lifespans1 = [sample_lifespan(material, region, rng1, config) for _ in range(10)]

    rng2 = RNGContext(seed=123).spawn(1)[0]
    lifespans2 = [sample_lifespan(material, region, rng2, config) for _ in range(10)]

    assert lifespans1 == lifespans2, "Lifespans should be identical for the same seed"

    # Verify different seed yields different results (probabilistically)
    rng3 = RNGContext(seed=456).spawn(1)[0]
    lifespans3 = [sample_lifespan(material, region, rng3, config) for _ in range(10)]
    assert lifespans1 != lifespans3, "Lifespans should differ for different seeds"


def test_lifespans_differ_across_regions_papyri(rng: np.random.Generator, config: SimulationConfig):
    """
    Verify that lifespans for Papyrus statistically differ across regions,
    specifically Egypt vs Asia Minor.
    Egypt papyri should on average live longer.
    """
    num_samples = 1000

    # Egypt Papyrus
    egypt_papyrus_lifespans = [sample_lifespan(Material.PAPYRUS, Region.EGYPT, rng, config) for _ in range(num_samples)]
    mean_egypt = np.mean(egypt_papyrus_lifespans)

    # Asia Minor Papyrus
    am_papyrus_lifespans = [sample_lifespan(Material.PAPYRUS, Region.ASIA_MINOR, rng, config) for _ in range(num_samples)]
    mean_am = np.mean(am_papyrus_lifespans)

    # Check for statistical difference: Egypt mean should be higher
    assert mean_egypt > mean_am, "Egypt papyri should statistically live longer than Asia Minor papyri."
    # Basic check for plausible range (not too strict)
    assert 50 < mean_am < 150
    assert 100 < mean_egypt < 300


def test_lifespans_differ_across_regions_parchment(rng: np.random.Generator, config: SimulationConfig):
    """
    Verify that lifespans for Parchment statistically differ across regions,
    specifically Egypt vs Asia Minor.
    Egypt parchment should on average live longer.
    """
    num_samples = 1000

    # Egypt Parchment
    egypt_parchment_lifespans = [sample_lifespan(Material.PARCHMENT, Region.EGYPT, rng, config) for _ in range(num_samples)]
    mean_egypt = np.mean(egypt_parchment_lifespans)

    # Asia Minor Parchment
    am_parchment_lifespans = [sample_lifespan(Material.PARCHMENT, Region.ASIA_MINOR, rng, config) for _ in range(num_samples)]
    mean_am = np.mean(am_parchment_lifespans)

    # Check for statistical difference: Egypt mean should be higher
    assert mean_egypt > mean_am, "Egypt parchment should statistically live longer than Asia Minor parchment."
    # Basic check for plausible range (not too strict)
    assert 100 < mean_am < 400
    assert 200 < mean_egypt < 800


def test_simulation_runs_without_death_ticks_config(config: SimulationConfig):
    """
    Verify that the simulation can run successfully without 'death_ticks'
    in the configuration, relying on the new probabilistic lifespan generation.
    """
    # Should not raise an error related to missing death_ticks
    rng = RNGContext(seed=123).spawn(1)[0]
    final_state = run_genealogy_generator(config, rng)

    # Basic assertion to ensure something ran
    assert final_state.tick == config.total_ticks
    assert len(final_state.registries.manuscripts) > 0
    assert len(final_state.registries.witnesses) > 0
    assert final_state.graph.number_of_nodes() > 0

    # Verify that death_ticks are indeed generated and applied
    for ms_id, manuscript in final_state.registries.manuscripts.items():
        assert manuscript.death_tick >= manuscript.birth_tick + 1
        assert manuscript.death_tick > manuscript.birth_tick


def test_death_tick_calculation_integration(config: SimulationConfig):
    """
    Verify that death_tick is correctly calculated as birth_tick + lifespan
    when a manuscript is spawned during a simulation run.
    """
    rng = RNGContext(seed=12345).spawn(1)[0]
    final_state = run_genealogy_generator(config, rng)

    # Check for each manuscript that death_tick > birth_tick, which implies a positive lifespan.
    for ms_id, manuscript in final_state.registries.manuscripts.items():
        assert manuscript.death_tick > manuscript.birth_tick

        sampled_lifespan_val = manuscript.death_tick - manuscript.birth_tick

        # Retrieve mu and sigma from the config
        material_name = manuscript.material.value
        region_name = manuscript.region.value
        params = config.lifespan_parameters[material_name][region_name]
        if isinstance(params, dict):
            mu, sigma = params["mu"], params["sigma"]
        else:
            mu, sigma = params.mu, params.sigma

        # Expected mean of the lognormal distribution is exp(mu + sigma^2 / 2)
        expected_mean_lifespan = np.exp(mu + sigma**2 / 2)

        # Set a very broad plausible range for the individual sampled lifespan
        assert 0.01 * expected_mean_lifespan < sampled_lifespan_val < 100 * expected_mean_lifespan
