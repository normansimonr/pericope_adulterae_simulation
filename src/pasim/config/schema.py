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
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, RootModel, model_validator, field_validator

from pasim.core.state import Region, Material, Script


class PersecutionEventConfig(BaseModel):
    """Configuration for a single persecution event."""
    start_tick: int = Field(..., ge=0)
    end_tick: Optional[int] = Field(None, ge=0)
    regions: List[str]
    kill_proportion: float = Field(..., ge=0.0, le=1.0)

    @field_validator('regions')
    @classmethod
    def validate_regions(cls, v):
        """Ensures all specified regions are valid."""
        valid_regions = {r.value for r in Region}
        for region_name in v:
            if region_name not in valid_regions:
                raise ValueError(f"Invalid region '{region_name}'. Must be one of {valid_regions}")
        return v

    @model_validator(mode='after')
    def validate_tick_ordering(self):
        """Ensures start_tick is not after end_tick."""
        if self.end_tick is not None and self.start_tick > self.end_tick:
            raise ValueError("start_tick cannot be after end_tick")
        return self

class MaterialTransitionConfig(BaseModel):
    """Configuration for a material transition point."""
    start_tick: int = Field(..., ge=0)
    distribution: Dict[str, float]

    @field_validator('distribution')
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

    @field_validator('distribution')
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

class DemandScheduleConfig(RootModel[Dict[int, Dict[str, int]]]):
    """Configuration for the demand schedule."""

    @model_validator(mode='after')
    def validate_demand_schedule(self):
        """Validates the demand schedule."""
        valid_regions = {r.value for r in Region}
        for tick, demands in self.root.items():
            if not isinstance(tick, int) or tick < 0:
                raise ValueError(f"Invalid tick '{tick}' in demand schedule. Must be a non-negative integer.")
            for region_name, count in demands.items():
                if region_name not in valid_regions:
                    raise ValueError(f"Invalid region '{region_name}' in demand schedule at tick {tick}. Must be one of {valid_regions}")
                if not isinstance(count, int) or count < 0:
                    raise ValueError(f"Invalid demand count '{count}' for region '{region_name}' at tick {tick}. Must be a non-negative integer.")
        return self

class SimulationConfig(BaseModel):
    """Root model for the entire simulation configuration."""
    total_ticks: int = Field(..., ge=1)
    p_region_migration: float = Field(0.0, ge=0.0, le=1.0)
    p_internal_relocation: float = Field(0.0, ge=0.0, le=1.0)
    reputation_distribution: List[float] = Field(..., min_length=5, max_length=5) # Assuming a 5-point distribution
    death_ticks: List[int]

    persecutions: List[PersecutionEventConfig] = []
    material_transitions: List[MaterialTransitionConfig] = []
    script_transitions: List[ScriptTransitionConfig] = []
    demand_schedule: DemandScheduleConfig

    @field_validator('material_transitions', 'script_transitions')
    @classmethod
    def validate_start_ticks_are_increasing(cls, v):
        """Ensures start_ticks in transition schedules are strictly increasing."""
        for i in range(len(v) - 1):
            if v[i].start_tick >= v[i+1].start_tick:
                raise ValueError("start_tick values in transition schedules must be strictly increasing.")
        return v
    
    @field_validator('reputation_distribution')
    @classmethod
    def validate_reputation_distribution_sum_to_one(cls, v):
        if not abs(sum(v) - 1.0) < 1e-9:
            raise ValueError(f"Reputation distribution probabilities must sum to 1.0, but got {sum(v)}")
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

def get_demand_for_tick(params: dict, tick: int) -> Dict[str, int]:
    """
    Returns the demand for a specific tick, using the last known value
    if the tick is not explicitly defined.
    """
    schedule = SimulationConfig(**params).demand_schedule.root
    
    if tick in schedule:
        return schedule[tick]
    
    # Find the last known tick before the current tick
    available_ticks = sorted([t for t in schedule.keys() if t < tick])
    if not available_ticks:
        return {}
        
    last_known_tick = available_ticks[-1]
    return schedule[last_known_tick]
