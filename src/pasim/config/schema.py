"""
Defines the comprehensive, hierarchical configuration schema for `pasim` simulations.

This module uses Pydantic to ensure that all simulation parameters are not only
structurally correct but also validated against logical rules (e.g., probabilities
summing to 1.0, start_ticks being sequential). It provides a single, authoritative
source of truth for what constitutes a valid experiment configuration.

The schema distinguishes between three primary kinds of historical drivers, which
are configured separately:

1.  **Historical Shocks (e.g., `persecutions`)**:
    These are discrete, instantaneous events that shock the system at a specific
    point in time. They represent external interventions, like a sudden persecution
    that destroys a fraction of manuscripts.

2.  **Environmental Regimes (e.g., `material_transitions`, `script_transitions`)**:
    These model long-term, gradual changes in the simulation's environment. They
    are defined as a schedule of time-dependent probability distributions that
    affect the properties of **newly created** entities, not existing ones. This
    is used to simulate cultural and technological shifts, such as the transition
    from papyrus to parchment or from uncial to minuscule script.

3.  **Structural Drivers (e.g., `demand_schedule`)**:
    These are core inputs that drive the fundamental mechanics of the simulation,
    such as the demand for new manuscripts in different regions over time. They are
    neither instantaneous shocks nor probabilistic environments but are a direct
    input to the simulation's core spawning logic.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator

from pasim.core.state import Material, Region, Script


class PersecutionEventConfig(BaseModel):
    """Configuration for a single persecution event."""

    start_tick: int = Field(..., ge=0)
    end_tick: Optional[int] = Field(None, ge=0)
    regions: List[str]
    kill_proportion: float = Field(..., ge=0.0, le=1.0)

    @field_validator("regions")
    @classmethod
    def validate_regions(cls, v):
        """Ensures all specified regions are valid."""
        valid_regions = {r.value for r in Region}
        for region_name in v:
            if region_name not in valid_regions:
                raise ValueError(f"Invalid region '{region_name}'. Must be one of {valid_regions}")
        return v

    @model_validator(mode="after")
    def validate_tick_ordering(self):
        """Ensures start_tick is not after end_tick."""
        if self.end_tick is not None and self.start_tick > self.end_tick:
            raise ValueError("start_tick cannot be after end_tick")
        return self


class MaterialTransitionConfig(BaseModel):
    """Configuration for a material transition point."""

    start_tick: int = Field(..., ge=0)
    distribution: Dict[str, float]

    @field_validator("distribution")
    @classmethod
    def validate_distribution_materials(cls, v):
        """Validates material names and ensures probabilities sum to 1."""
        valid_materials = {m.value for m in Material}
        total_prob = 0.0
        for material_name, prob in v.items():
            if material_name not in valid_materials:
                raise ValueError(f"Invalid material '{material_name}'. Must be one of {valid_materials}")
            if prob < 0:
                raise ValueError(f"Probability for '{material_name}' cannot be negative.")
            total_prob += prob

        if not abs(total_prob - 1.0) < 1e-9:
            raise ValueError(f"Probabilities must sum to 1.0, but got {total_prob}")
        return v


class ScriptTransitionConfig(BaseModel):
    """Configuration for a script transition point."""

    start_tick: int = Field(..., ge=0)
    distribution: Dict[str, float]

    @field_validator("distribution")
    @classmethod
    def validate_distribution_scripts(cls, v):
        """Validates script names and ensures probabilities sum to 1."""
        valid_scripts = {s.value for s in Script}
        total_prob = 0.0
        for script_name, prob in v.items():
            if script_name not in valid_scripts:
                raise ValueError(f"Invalid script '{script_name}'. Must be one of {valid_scripts}")
            if prob < 0:
                raise ValueError(f"Probability for '{script_name}' cannot be negative.")
            total_prob += prob

        if not abs(total_prob - 1.0) < 1e-9:
            raise ValueError(f"Probabilities must sum to 1.0, but got {total_prob}")
        return v


class DemandScheduleConfig(RootModel[Dict[int, int]]):
    """Configuration for the demand schedule."""

    @model_validator(mode="after")
    def validate_demand_schedule(self):
        """Validates the demand schedule after region strings have been converted to enums."""
        for tick, aggregate_demand in self.root.items():
            if not isinstance(tick, int) or tick < 0:
                raise ValueError(f"Invalid tick '{tick}' in demand schedule. Must be a non-negative integer.")
            if not isinstance(aggregate_demand, int) or aggregate_demand < 0:
                raise ValueError(f"Invalid aggregate demand count '{aggregate_demand}' for tick {tick}. Must be a non-negative integer.")
        return self


class LifespanConfig(BaseModel):
    """Configuration for lognormal lifespan parameters."""

    mu: float
    sigma: float


class SimulationConfig(BaseModel):
    """Root model for the entire simulation configuration."""

    total_ticks: int = Field(..., ge=1)
    text_length: int = Field(200, ge=1)
    p_region_migration: float = Field(0.0, ge=0.0, le=1.0)
    p_internal_relocation: float = Field(0.0, ge=0.0, le=1.0)
    p_uncial_exemplar_death_on_minuscule_birth: float = Field(0.0, ge=0.0, le=1.0)
    reputation_distribution: Dict[int, float]

    # Lifespan parameters (Material -> Region -> LifespanConfig)
    lifespan_parameters: Dict[str, Dict[str, LifespanConfig]] = Field(
        default={
            "papyrus": {
                "Asia Minor": {"mu": 4.72, "sigma": 0.30},
                "Levant": {"mu": 4.72, "sigma": 0.30},
                "Egypt": {"mu": 5.12, "sigma": 0.60},
            },
            "parchment": {
                "Asia Minor": {"mu": 5.5, "sigma": 0.5},
                "Levant": {"mu": 5.5, "sigma": 0.5},
                "Egypt": {"mu": 6.0, "sigma": 0.5},
            },
            "paper": {
                "Asia Minor": {"mu": 5.0, "sigma": 0.5},
                "Levant": {"mu": 5.0, "sigma": 0.5},
                "Egypt": {"mu": 5.5, "sigma": 0.5},
            },
        }
    )

    # Region bounds (Region -> [[xmin, xmax], [ymin, ymax]])
    region_bounds: Dict[str, List[List[float]]] = Field(
        default={
            "Asia Minor": [[0.0, 100.0], [0.0, 100.0]],
            "Egypt": [[200.0, 300.0], [200.0, 300.0]],
            "Levant": [[400.0, 500.0], [400.0, 500.0]],
        }
    )

    # New configurable mappings and targets
    reputation_mutation_mapping: Dict[int, float] = Field(
        default={
            1: 0.10,
            2: 0.10,
            3: 0.30,
            4: 0.30,
            5: 0.20,
        }
    )

    survivor_sampling_targets: Dict[str, Any] = Field(
        default={
            "target_total": 3000,
            "target_born_650_or_later": 2400,
            "target_born_earlier": 600,
            "proportion_asia_minor": 0.98,
            "sampling_seed_offset": 999999,
            "eligibility_min_tick": 300,
            "strata_boundary_tick": 650,
            "strata_focus_region": "Asia Minor",
        }
    )

    regional_demand_distributions: Dict[str, Dict[str, float]] = Field(
        default={
            "0-2": {"Asia Minor": 0.70, "Levant": 0.25, "Egypt": 0.05},
            "3-5": {"Asia Minor": 0.55, "Levant": 0.25, "Egypt": 0.20},
            "6+": {"Asia Minor": 1.00, "Levant": 0.00, "Egypt": 0.00},
        }
    )

    parent_num_distribution: List[float] = Field(
        default=[0.8, 0.1, 0.1],  # Probabilities for 1, 2, 3 parents
        min_length=3,
        max_length=3,
    )

    geographical_candidate_pool_size: int = Field(10, ge=1)
    pa_intervention_radius: float = Field(10.0, ge=0.0)

    persecutions: List[PersecutionEventConfig] = []
    material_transitions: List[MaterialTransitionConfig] = []
    script_transitions: List[ScriptTransitionConfig] = []
    demand_schedule: DemandScheduleConfig
    log_tick_frequency: int = Field(100, ge=1)
    validation_frequency: int = Field(10, ge=0)  # 0 means disabled, N means every N ticks

    # PA Regime Configuration
    pa_regime: Literal["insertion", "omission"]
    pa_intervention_year: int = Field(..., ge=0)
    pa_intervention_region: Region
    pa_innovator_reputation: float = Field(..., ge=1.0, le=5.0)

    # Execution / Persistence Configuration
    persistence_level: Literal["minimal", "full"] = "minimal"

    @field_validator("parent_num_distribution")
    @classmethod
    def validate_parent_dist(cls, v):
        if not abs(sum(v) - 1.0) < 1e-9:
            raise ValueError("parent_num_distribution must sum to 1.0")
        return v

    @model_validator(mode="after")
    def validate_pa_intervention_year(self):
        """Ensures pa_intervention_year is within simulation total_ticks."""
        if self.pa_intervention_year > self.total_ticks:
            raise ValueError(f"pa_intervention_year ({self.pa_intervention_year}) cannot exceed total_ticks ({self.total_ticks})")
        return self

    @field_validator("material_transitions", "script_transitions")
    @classmethod
    def validate_start_ticks_are_increasing(cls, v):
        """Ensures start_ticks in transition schedules are strictly increasing."""
        for i in range(len(v) - 1):
            if v[i].start_tick >= v[i + 1].start_tick:
                raise ValueError("start_tick values in transition schedules must be strictly increasing.")
        return v

    @field_validator("reputation_distribution")
    @classmethod
    def validate_reputation_distribution(cls, v: Dict[int, float]):
        """
        Validates the reputation distribution: keys must be 1-5, values must be non-negative,
        and probabilities must sum to 1.0.
        """
        expected_keys = {1, 2, 3, 4, 5}
        if set(v.keys()) != expected_keys:
            raise ValueError(f"Reputation distribution keys must be 1-5, but got {set(v.keys())}")

        total_prob = 0.0
        for rep_score, prob in v.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Reputation probability for score {rep_score} must be between 0.0 and 1.0, got {prob}")
            total_prob += prob

        if not abs(total_prob - 1.0) < 1e-9:
            raise ValueError(f"Reputation distribution probabilities must sum to 1.0, but got {total_prob}")
        return v


def get_persecution_events(params: dict) -> List[dict]:
    """Returns the validated persecution events."""
    return [p.model_dump() for p in SimulationConfig(**params).persecutions]


def get_material_schedule(params: dict) -> List[dict]:
    """Returns the validated material transition schedule."""
    return [m.model_dump() for m in SimulationConfig(**params).material_transitions]


def get_script_schedule(params: dict) -> List[dict]:
    """Returns the validated script transition schedule."""
    return [s.model_dump() for s in SimulationConfig(**params).script_transitions]
