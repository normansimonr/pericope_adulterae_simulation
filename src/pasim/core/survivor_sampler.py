import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from pasim.config.schema import SimulationConfig
from pasim.core.genealogy_snapshot import GenealogySnapshot
from pasim.core.state import Region


@dataclass
class SamplingResult:
    """Results of the survivorship sampling process."""

    sampled_witness_ids: List[str]
    actual_sample_size: int
    stratum_counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)
    deficit_per_stratum: Dict[str, int] = field(default_factory=dict)


def sample_survivors(snapshot: GenealogySnapshot, base_seed: int, total_ticks: int, config: SimulationConfig) -> SamplingResult:
    """
    Deterministically samples a subset of witnesses based on survivorship bias.
    """
    targets = config.survivor_sampling_targets
    sampling_seed_offset = targets.get("sampling_seed_offset", 999999)
    target_total = targets.get("target_total", 3000)
    target_650_plus = targets.get("target_born_650_or_later", 2400)
    target_born_earlier = targets.get("target_born_earlier", 600)
    proportion_am = targets.get("proportion_asia_minor", 0.98)

    eligibility_min_tick = targets.get("eligibility_min_tick", 300)
    strata_boundary_tick = targets.get("strata_boundary_tick", 650)
    strata_focus_region_name = targets.get("strata_focus_region", "Asia Minor")
    strata_focus_region = Region(strata_focus_region_name)

    sampling_seed = (base_seed + sampling_seed_offset) & 0xFFFFFFFF
    rng = np.random.default_rng(sampling_seed)

    # 1. Filter eligible candidates
    eligible_nodes = [node for node in snapshot.nodes if node.birth_tick >= eligibility_min_tick and node.death_tick >= total_ticks]

    if not eligible_nodes:
        return SamplingResult(
            sampled_witness_ids=[], actual_sample_size=0, stratum_counts={}, warnings=["No eligible nodes found for survivorship sampling."]
        )

    # 2. Categorize into strata
    # Strata definitions:
    # 1: FocusRegion, born >= Boundary
    # 2: FocusRegion, Eligibility <= born < Boundary
    # 3: Other, born >= Boundary
    # 4: Other, Eligibility <= born < Boundary

    s1_name = f"{strata_focus_region_name}_{strata_boundary_tick}+"
    s2_name = f"{strata_focus_region_name}_{eligibility_min_tick}-{strata_boundary_tick - 1}"
    s3_name = f"Other_{strata_boundary_tick}+"
    s4_name = f"Other_{eligibility_min_tick}-{strata_boundary_tick - 1}"

    strata: Dict[str, List[str]] = {s1_name: [], s2_name: [], s3_name: [], s4_name: []}

    for node in eligible_nodes:
        is_focus_region = node.region == strata_focus_region
        is_boundary_plus = node.birth_tick >= strata_boundary_tick

        if is_focus_region:
            if is_boundary_plus:
                strata[s1_name].append(node.instance_id)
            else:
                strata[s2_name].append(node.instance_id)
        else:
            if is_boundary_plus:
                strata[s3_name].append(node.instance_id)
            else:
                strata[s4_name].append(node.instance_id)

    # 3. Determine target allocation
    t1 = int(target_total * proportion_am * (target_650_plus / target_total))
    t2 = int(target_total * proportion_am * (target_born_earlier / target_total))
    t3 = int(target_total * (1 - proportion_am) * (target_650_plus / target_total))
    t4 = int(target_total * (1 - proportion_am) * (target_born_earlier / target_total))

    target_alloc = {s1_name: t1, s2_name: t2, s3_name: t3, s4_name: t4}

    actual_alloc: Dict[str, List[str]] = {}
    sampled_ids: Set[str] = set()
    warnings = []
    deficits = {}

    # 4. Perform sampling per stratum
    # Order: Other first, then Focus Region.
    ordered_strata = [s3_name, s4_name, s2_name, s1_name]

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
                if s_name == s3_name:
                    current_targets[s1_name] += deficit
                elif s_name == s4_name:
                    current_targets[s2_name] += deficit
                elif s_name == s2_name:
                    current_targets[s1_name] += deficit
        else:
            # Sample randomly but deterministically
            indices = rng.choice(len(pool), size=target, replace=False)
            selected = [pool[i] for i in indices]

        actual_alloc[s_name] = selected
        sampled_ids.update(selected)

    stratum_counts = {k: len(v) for k, v in actual_alloc.items()}
    actual_sample_size = len(sampled_ids)

    if actual_sample_size < target_total:
        warnings.append(f"Could not reach target sample size of {target_total}. Actual: {actual_sample_size}")

    return SamplingResult(
        sampled_witness_ids=sorted(list(sampled_ids)),
        actual_sample_size=actual_sample_size,
        stratum_counts=stratum_counts,
        warnings=warnings,
        deficit_per_stratum=deficits,
    )


def save_sampling_results(result: SamplingResult, run_dir: Path, target_total: int):
    """Saves survivors.json and sampling_log.json."""
    survivors_data = {"sample_size": result.actual_sample_size, "sampled_witness_ids": result.sampled_witness_ids}
    with open(run_dir / "survivors.json", "w") as f:
        json.dump(survivors_data, f, indent=2)

    log_data = {
        "target_sample_size": target_total,
        "actual_sample_size": result.actual_sample_size,
        "stratum_counts": result.stratum_counts,
        "warnings": result.warnings,
        "deficit_per_stratum": result.deficit_per_stratum,
    }
    with open(run_dir / "sampling_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
