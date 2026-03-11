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
    targets_config = config.survivor_sampling_targets
    sampling_seed_offset = targets_config.get("sampling_seed_offset", 999999)
    target_total = targets_config.get("target_total", 3000)

    sampling_seed = (base_seed + sampling_seed_offset) & 0xFFFFFFFF
    rng = np.random.default_rng(sampling_seed)

    # 1. Filter eligible candidates
    eligibility_min_tick = targets_config.get("eligibility_min_tick", 300)
    eligible_nodes = [node for node in snapshot.nodes if node.birth_tick >= eligibility_min_tick and node.death_tick >= total_ticks]

    if not eligible_nodes:
        return SamplingResult(
            sampled_witness_ids=[], actual_sample_size=0, stratum_counts={}, warnings=["No eligible nodes found for survivorship sampling."]
        )

    # 2. Categorize into strata
    strata_names = _get_strata_names(targets_config)
    strata = _categorize_into_strata(eligible_nodes, targets_config, strata_names)

    # 3. Determine target allocation
    target_alloc = _calculate_target_allocation(target_total, targets_config, strata_names)

    # 4. Perform sampling per stratum
    # Order: Other first, then Focus Region.
    ordered_strata = [strata_names["s3"], strata_names["s4"], strata_names["s2"], strata_names["s1"]]
    sampled_ids, stratum_counts, warnings, deficits = _sample_from_strata(strata, target_alloc, rng, ordered_strata)

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


def _get_strata_names(targets_config: dict) -> Dict[str, str]:
    """Generates names for the sampling strata."""
    strata_focus_region_name = targets_config.get("strata_focus_region", "Asia Minor")
    strata_boundary_tick = targets_config.get("strata_boundary_tick", 650)
    eligibility_min_tick = targets_config.get("eligibility_min_tick", 300)

    return {
        "s1": f"{strata_focus_region_name}_{strata_boundary_tick}+",
        "s2": f"{strata_focus_region_name}_{eligibility_min_tick}-{strata_boundary_tick - 1}",
        "s3": f"Other_{strata_boundary_tick}+",
        "s4": f"Other_{eligibility_min_tick}-{strata_boundary_tick - 1}",
    }


def _categorize_into_strata(eligible_nodes: list, targets_config: dict, names: Dict[str, str]) -> Dict[str, List[str]]:
    """Groups eligible nodes into geographical and temporal strata."""
    strata_focus_region = Region(targets_config.get("strata_focus_region", "Asia Minor"))
    strata_boundary_tick = targets_config.get("strata_boundary_tick", 650)

    strata: Dict[str, List[str]] = {name: [] for name in names.values()}

    for node in eligible_nodes:
        is_focus_region = node.region == strata_focus_region
        is_boundary_plus = node.birth_tick >= strata_boundary_tick

        if is_focus_region:
            dest = names["s1"] if is_boundary_plus else names["s2"]
        else:
            dest = names["s3"] if is_boundary_plus else names["s4"]
        strata[dest].append(node.instance_id)

    return strata


def _calculate_target_allocation(target_total: int, targets_config: dict, names: Dict[str, str]) -> Dict[str, int]:
    """Calculates the target number of witnesses to sample from each stratum."""
    target_650_plus = targets_config.get("target_born_650_or_later", 2400)
    target_born_earlier = targets_config.get("target_born_earlier", 600)
    proportion_am = targets_config.get("proportion_asia_minor", 0.98)

    t1 = int(target_total * proportion_am * (target_650_plus / target_total))
    t2 = int(target_total * proportion_am * (target_born_earlier / target_total))
    t3 = int(target_total * (1 - proportion_am) * (target_650_plus / target_total))
    t4 = int(target_total * (1 - proportion_am) * (target_born_earlier / target_total))

    return {names["s1"]: t1, names["s2"]: t2, names["s3"]: t3, names["s4"]: t4}


def _sample_from_strata(
    strata: Dict[str, List[str]],
    target_alloc: Dict[str, int],
    rng: np.random.Generator,
    ordered_strata: List[str],
) -> tuple[Set[str], Dict[str, int], List[str], Dict[str, int]]:
    """Performs the actual random sampling from each stratum with redistribution of deficits."""
    current_targets = target_alloc.copy()
    actual_alloc: Dict[str, List[str]] = {}
    sampled_ids: Set[str] = set()
    warnings = []
    deficits = {}

    for s_name in ordered_strata:
        pool = strata[s_name]
        target = current_targets[s_name]

        if len(pool) <= target:
            selected = pool
            if len(pool) < target:
                deficit = target - len(pool)
                deficits[s_name] = deficit
                warnings.append(f"Stratum {s_name} insufficient: target {target}, available {len(pool)}")
                _redistribute_sampling_deficit(s_name, deficit, current_targets, ordered_strata)
        else:
            indices = rng.choice(len(pool), size=target, replace=False)
            selected = [pool[i] for i in indices]

        actual_alloc[s_name] = selected
        sampled_ids.update(selected)

    stratum_counts = {k: len(v) for k, v in actual_alloc.items()}
    return sampled_ids, stratum_counts, warnings, deficits


def _redistribute_sampling_deficit(s_name: str, deficit: int, current_targets: Dict[str, int], ordered_strata: List[str]):
    """Redistributes sampling deficits from exhausted strata to others."""
    # ordered_strata = [s3_name, s4_name, s2_name, s1_name]
    s3_name, s4_name, s2_name, s1_name = ordered_strata
    if s_name == s3_name:
        current_targets[s1_name] += deficit
    elif s_name == s4_name:
        current_targets[s2_name] += deficit
    elif s_name == s2_name:
        current_targets[s1_name] += deficit


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
