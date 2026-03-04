import numpy as np

from pasim.config.schema import SimulationConfig


def make_initial_text(config: SimulationConfig) -> np.ndarray:
    """
    Create the base tagged string for the autograph.

    This creates a simple text of a fixed length, defined in the simulation
    configuration, with all segment values initialised to zero.

    Args:
        config: The validated simulation configuration object.

    Returns:
        A numpy array of int16 zeros representing the initial text.
    """
    length = config.text_length
    return np.zeros(length, dtype=np.int16)
