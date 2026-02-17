"""
This module provides functions for probabilistically generating manuscript lifespans
based on their material and region.
"""

from typing import Tuple

import numpy as np

from pasim.core.state import Material, Region
from pasim.core.rng import RNG


# Define lognormal distribution parameters (mu, sigma) for different materials and regions
# These parameters are for log(lifespan)
LOGNORMAL_PARAMETERS = {
    Material.PAPYRUS: {
        (Region.ASIA_MINOR, Region.LEVANT): {"mu": 4.72, "sigma": 0.30}, # Changed PALESTINE to LEVANT
        Region.EGYPT: {"mu": 5.12, "sigma": 0.60},
    },
    Material.PARCHMENT: {
        (Region.ASIA_MINOR, Region.LEVANT): {"mu": 5.5, "sigma": 0.5}, # Changed PALESTINE to LEVANT
        Region.EGYPT: {"mu": 6.0, "sigma": 0.5},
    },
    Material.PAPER: { # Added PAPER parameters
        (Region.ASIA_MINOR, Region.LEVANT): {"mu": 5.0, "sigma": 0.5},
        Region.EGYPT: {"mu": 5.5, "sigma": 0.5},
    },
}

def _get_lognormal_params(material: Material, region: Region) -> Tuple[float, float]:
    """
    Retrieves the lognormal distribution parameters (mu, sigma) for a given
    material and region.
    """
    material_params = LOGNORMAL_PARAMETERS.get(material)
    if material_params is None:
        raise ValueError(f"Unsupported material for lifespan calculation: {material.value}")

    # Check for direct region match
    region_params = material_params.get(region)
    if region_params is not None:
        return region_params["mu"], region_params["sigma"]

    # Check for region groups (e.g., ASIA_MINOR and PALESTINE)
    for region_group, params in material_params.items():
        if isinstance(region_group, tuple) and region in region_group:
            return params["mu"], params["sigma"]

    raise ValueError(f"Unsupported region for lifespan calculation: {region.value} with material {material.value}")


def sample_lifespan(material: Material, region: Region, rng: np.random.Generator) -> int:
    """
    Samples a manuscript lifespan in years (simulation ticks) from a lognormal
    distribution based on its material and region.

    Args:
        material (Material): The material of the manuscript.
        region (Region): The region where the manuscript is located.
        rng (np.random.Generator): The NumPy random number generator for sampling.

    Returns:
        int: The sampled lifespan in years, rounded to an integer,
             and guaranteed to be at least 1.
    """
    mu, sigma = _get_lognormal_params(material, region)

    # Sample from lognormal distribution
    lifespan_float = rng.lognormal(mean=mu, sigma=sigma)

    # Convert to integer years, with a minimum of 1
    lifespan_int = max(1, int(np.floor(lifespan_float)))

    return lifespan_int
