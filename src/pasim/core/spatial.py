"""
Utilities for handling spatial assignments of manuscripts.

This module provides functions for assigning region-specific coordinates to
manuscripts. It enforces the architectural principle that spatial properties
belong to the physical manuscript and not to the abstract genealogy graph.
"""
from typing import Tuple
from numpy.random import Generator as RNG

from pasim.core.state import Region

# Bounding boxes for each geographical region, defining independent planar spaces.
# Coordinates are not comparable across regions.
# Format: {Region: ((xmin, xmax), (ymin, ymax))}
REGION_BOUNDS = {
    Region.ASIA_MINOR: ((0.0, 100.0), (0.0, 100.0)),
    Region.EGYPT: ((200.0, 300.0), (200.0, 300.0)),
    Region.LEVANT: ((400.0, 500.0), (400.0, 500.0)),
}


def generate_random_coordinates(region: Region, rng: RNG) -> Tuple[float, float]:
    """Generates a random (x, y) coordinate within the bounds of a given region.

    This function ensures that manuscript locations are deterministically generated
    and scoped to their assigned region's independent planar space.

    Args:
        region: The geographical region for which to generate coordinates.
        rng: The seeded random number generator to ensure reproducibility.

    Returns:
        A tuple containing the (x, y) coordinates.

    Raises:
        ValueError: If the provided region is not defined in REGION_BOUNDS.
    """
    if region not in REGION_BOUNDS:
        raise ValueError(f"Region '{region}' does not have defined coordinate bounds.")

    x_bounds, y_bounds = REGION_BOUNDS[region]
    x = rng.uniform(x_bounds[0], x_bounds[1])
    y = rng.uniform(y_bounds[0], y_bounds[1])

    return x, y
