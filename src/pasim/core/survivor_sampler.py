import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from pasim.core.genealogy_snapshot import GenealogySnapshot
from pasim.core.state import Region

SAMPLING_SEED_OFFSET = 999999
TARGET_TOTAL = 3000
TARGET_BORN_650_OR_LATER = 2400
TARGET_BORN_EARLIER = 600
PROPORTION_ASIA_MINOR = 0.98


@dataclass
class SamplingResult:
    """Results of the survivorship sampling process."""

    sampled_witness_ids: List[str]
    actual_sample_size: int
    stratum_counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)
    deficit_per_stratum: Dict[str, int] = field(default_factory=dict)


def sample_survivors(snapshot: GenealogySnapshot, base_seed: int, total_ticks: int) -> SamplingResult:
    """
    Deterministically samples a subset of witnesses based on survivorship bias.

    Eligibility:
    - born >= 300
    - death_tick >= total_ticks (alive at end)
    """
    sampling_seed = (base_seed + SAMPLING_SEED_OFFSET) & 0xFFFFFFFF
    rng = np.random.default_rng(sampling_seed)

    # 1. Filter eligible candidates
    eligible_nodes = [node for node in snapshot.nodes if node.birth_tick >= 300 and node.death_tick >= total_ticks]

    if not eligible_nodes:
        return SamplingResult(
            sampled_witness_ids=[], actual_sample_size=0, stratum_counts={}, warnings=["No eligible nodes found for survivorship sampling."]
        )

    # 2. Categorize into strata
    # Strata definitions:
    # 1: AsiaMinor, born >= 650
    # 2: AsiaMinor, 300 <= born < 650
    # 3: Other, born >= 650
    # 4: Other, 300 <= born < 650

    strata: Dict[str, List[str]] = {"AsiaMinor_650+": [], "AsiaMinor_300-649": [], "Other_650+": [], "Other_300-649": []}

    for node in eligible_nodes:
        is_asia_minor = node.region == Region.ASIA_MINOR
        is_650_plus = node.birth_tick >= 650

        if is_asia_minor:
            if is_650_plus:
                strata["AsiaMinor_650+"].append(node.instance_id)
            else:
                strata["AsiaMinor_300-649"].append(node.instance_id)
        else:
            if is_650_plus:
                strata["Other_650+"].append(node.instance_id)
            else:
                strata["Other_300-649"].append(node.instance_id)

    # 3. Determine target allocation (from prompt example)
    # Target: 3000 total
    # Born 650+: 2400 (Strata 1 + 3)
    # Born 300-649: 600 (Strata 2 + 4)
    # Asia Minor: 2940 (Strata 1 + 2)
    # Other: 60 (Strata 3 + 4)
    #
    # Solution satisfying all (if enough population):
    # S1 (AM 650+): 2340
    # S2 (AM 300-649): 600
    # S3 (Other 650+): 60
    # S4 (Other 300-649): 0

    target_alloc = {"AsiaMinor_650+": 2340, "AsiaMinor_300-649": 600, "Other_650+": 60, "Other_300-649": 0}

    actual_alloc: Dict[str, List[str]] = {}
    sampled_ids: Set[str] = set()
    warnings = []
    deficits = {}

    # 4. Perform sampling per stratum
    # We do it in an order that allows redistribution if needed.
    # Order: Other first, then Asia Minor.

    ordered_strata = ["Other_650+", "Other_300-649", "AsiaMinor_300-649", "AsiaMinor_650+"]

    current_targets = target_alloc.copy()

    for s_name in ordered_strata:
        pool = strata[s_name]
        target = current_targets[s_name]

        if len(pool) <= target:
            # Take all available
            selected = pool
            if len(pool) < target:
                deficit = target - len(pool)
                deficits[s_name] = deficit
                warnings.append(f"Stratum {s_name} insufficient: target {target}, available {len(pool)}")
                # Redistribution logic:
                # If Other_650+ is short, try to take more from AsiaMinor_650+?
                # Prompt says: "redistribute remaining quota to compatible strata where possible"
                # Compatibility: Same time bracket preferred.
                if s_name == "Other_650+":
                    current_targets["AsiaMinor_650+"] += deficit
                elif s_name == "Other_300-649":
                    current_targets["AsiaMinor_300-649"] += deficit
                elif s_name == "AsiaMinor_300-649":
                    current_targets["AsiaMinor_650+"] += deficit
                # If AsiaMinor_650+ is short, there's no "compatible" one left in this simple redistribution.
        else:
            # Sample randomly but deterministically
            indices = rng.choice(len(pool), size=target, replace=False)
            selected = [pool[i] for i in indices]

        actual_alloc[s_name] = selected
        sampled_ids.update(selected)

    stratum_counts = {k: len(v) for k, v in actual_alloc.items()}
    actual_sample_size = len(sampled_ids)

    if actual_sample_size < TARGET_TOTAL:
        warnings.append(f"Could not reach target sample size of {TARGET_TOTAL}. Actual: {actual_sample_size}")

    return SamplingResult(
        sampled_witness_ids=sorted(list(sampled_ids)),
        actual_sample_size=actual_sample_size,
        stratum_counts=stratum_counts,
        warnings=warnings,
        deficit_per_stratum=deficits,
    )


def save_sampling_results(result: SamplingResult, run_dir: Path):
    """Saves survivors.json and sampling_log.json."""
    survivors_data = {"sample_size": result.actual_sample_size, "sampled_witness_ids": result.sampled_witness_ids}
    with open(run_dir / "survivors.json", "w") as f:
        json.dump(survivors_data, f, indent=2)

    log_data = {
        "target_sample_size": TARGET_TOTAL,
        "actual_sample_size": result.actual_sample_size,
        "stratum_counts": result.stratum_counts,
        "warnings": result.warnings,
        "deficit_per_stratum": result.deficit_per_stratum,
    }
    with open(run_dir / "sampling_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
