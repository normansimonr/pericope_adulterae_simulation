"""
This module provides utility functions for managing spatial assignments of
manuscripts within the simulation.

Key Principles:
- Geographical properties are associated with the physical `Manuscript` objects,
  not the abstract genealogy nodes.
- Regions are independent planar (x, y) coordinate spaces. Coordinates are not
  comparable across regions (e.g., (50, 50) in Asia Minor is not geographically
  related to (50, 50) in Egypt).
"""

from typing import Tuple

import numpy as np

from pasim.config.schema import SimulationConfig
from pasim.core.state import Region


def generate_random_coordinates(region: Region, rng: np.random.Generator, config: SimulationConfig) -> Tuple[float, float]:
    """
    Generates a random (x, y) coordinate within the bounds of a given region,
    using boundaries defined in the simulation configuration.

    Args:
        region (Region): The geographical region.
        rng (np.random.Generator): The random number generator.
        config (SimulationConfig): The simulation configuration containing region bounds.

    Returns:
        Tuple[float, float]: The generated (x, y) coordinates.
    """
    region_name = region.value
    bounds = config.region_bounds.get(region_name)

    if bounds is None:
        raise ValueError(f"No bounds defined for region: {region_name}")

    x_bounds = bounds[0]
    y_bounds = bounds[1]

    x = rng.uniform(x_bounds[0], x_bounds[1])
    y = rng.uniform(y_bounds[0], y_bounds[1])

    return (x, y)
