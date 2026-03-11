"""
This module provides functions for probabilistically generating manuscript lifespans
based on their material and region.
"""

from typing import Tuple

import numpy as np

from pasim.config.schema import SimulationConfig
from pasim.core.state import Material, Region


def _get_lognormal_params(material: Material, region: Region, config: SimulationConfig) -> Tuple[float, float]:
    """
    Retrieves the lognormal distribution parameters (mu, sigma) for a given
    material and region from the simulation configuration.
    """
    material_name = material.value
    region_name = region.value

    lifespan_params = config.lifespan_parameters

    if material_name not in lifespan_params:
        raise ValueError(f"Unsupported material for lifespan calculation: {material_name}")

    material_config = lifespan_params[material_name]

    if region_name not in material_config:
        raise ValueError(f"Unsupported region for lifespan calculation: {region_name} with material {material_name}")

    params = material_config[region_name]
    if isinstance(params, dict):
        return params["mu"], params["sigma"]
    return params.mu, params.sigma


def sample_lifespan(material: Material, region: Region, rng: np.random.Generator, config: SimulationConfig) -> int:
    """
    Samples a manuscript lifespan in years (simulation ticks) from a lognormal
    distribution based on its material and region, using parameters from config.

    Args:
        material (Material): The material of the manuscript.
        region (Region): The region where the manuscript is located.
        rng (np.random.Generator): The NumPy random number generator for sampling.
        config (SimulationConfig): The simulation configuration.

    Returns:
        int: The sampled lifespan in years, rounded to an integer,
             and guaranteed to be at least 1.
    """
    mu, sigma = _get_lognormal_params(material, region, config)

    # Sample from lognormal distribution
    lifespan_float = rng.lognormal(mean=mu, sigma=sigma)

    # Convert to integer years, with a minimum of 1
    lifespan_int = max(1, int(np.floor(lifespan_float)))

    return lifespan_int
