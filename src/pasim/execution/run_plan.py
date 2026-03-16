from dataclasses import dataclass
from typing import List, Optional

from pasim.core.rng import RNGContext


@dataclass(frozen=True)
class RunSpec:
    """
    Defines the specification for a single simulation run.
    """

    run_id: int
    seed: int


def generate_run_plan(n_runs: int, seed: Optional[int] = None, skip_run_ids: Optional[List[int]] = None) -> List[RunSpec]:
    """
    Generates a deterministic list of RunSpec objects.

    Args:
        n_runs: The number of runs to plan.
        seed: The master seed for generating individual run seeds.
        skip_run_ids: Optional list of run_ids to exclude from the final plan.

    Returns:
        A list of RunSpec objects.
    """
    rng_context = RNGContext(seed=seed)
    # Generate a unique seed for each run using the same logic as before
    # We MUST generate all seeds to maintain determinism for future runs
    run_seeds = [int(gen.integers(0, 2**32 - 1)) for gen in rng_context.spawn(n_runs)]

    skip_set = set(skip_run_ids or [])

    return [RunSpec(run_id=i, seed=run_seeds[i]) for i in range(n_runs) if i not in skip_set]
