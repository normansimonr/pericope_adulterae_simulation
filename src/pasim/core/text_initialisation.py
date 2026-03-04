import numpy as np

from pasim.config.schema import SimulationConfig


def make_initial_text(config: SimulationConfig) -> np.ndarray:
    """
    Create the base tagged string for the autograph based on the PA regime.

    If the regime is "insertion", the initial genome is all 0.
    If the regime is "omission", the initial genome is all 1.

    Args:
        config: The validated simulation configuration object.

    Returns:
        A numpy array of int16 values representing the initial text.
    """
    length = config.text_length
    if config.pa_regime == "insertion":
        return np.zeros(length, dtype=np.int16)
    elif config.pa_regime == "omission":
        return np.ones(length, dtype=np.int16)
    else:
        # Pydantic validation should prevent this, but for safety:
        raise ValueError(f"Unknown PA regime: {config.pa_regime}")
