from pasim.core.genealogy_generator import run_genealogy_generator
from pasim.core.rng import RNGContext
from pasim.core.state import DeathReason, Script


def test_uncial_death_on_minuscule_birth():
    """Verify that Uncial parents can be killed when Minuscule children are born."""
    # Define a simple experiment where we have Uncials and then Minuscule is born
    params = {
        "total_ticks": 10,
        "n_runs": 1,
        "seed": 42,
        "text_length": 10,
        "p_region_migration": 0.0,
        "p_internal_relocation": 0.0,
        "p_uncial_exemplar_death_on_minuscule_birth": 1.0,  # 100% chance for testing
        "reputation_distribution": {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        "pa_regime": "insertion",
        "pa_intervention_year": 1,  # Valid tick, but won't trigger innovator at birth 0/5
        "pa_intervention_region": "Asia Minor",
        "pa_innovator_reputation": 5.0,
        "demand_schedule": {
            0: 1,  # Spawn an Uncial
            5: 2,  # Spawn a Minuscule that copies the Uncial
        },
        "material_transitions": [{"start_tick": 0, "distribution": {"parchment": 1.0}}],
        "script_transitions": [{"start_tick": 0, "distribution": {"uncial": 1.0}}, {"start_tick": 5, "distribution": {"minuscule": 1.0}}],
        "lifespan_parameters": {
            "parchment": {
                "Asia Minor": {"mu": 10.0, "sigma": 0.1},  # Long lifespan
                "Egypt": {"mu": 10.0, "sigma": 0.1},
                "Levant": {"mu": 10.0, "sigma": 0.1},
            }
        },
        "regional_demand_distributions": {
            "0-2": {"Asia Minor": 1.0, "Levant": 0.0, "Egypt": 0.0},
            "3-5": {"Asia Minor": 1.0, "Levant": 0.0, "Egypt": 0.0},
            "6+": {"Asia Minor": 1.0, "Levant": 0.0, "Egypt": 0.0},
        },
    }

    rng_context = RNGContext(seed=42)
    rng = rng_context.spawn(1)[0]

    state = run_genealogy_generator(params, rng)

    print("\nManuscripts in registry:")
    for mid, m in state.registries.manuscripts.items():
        w = state.registries.witnesses.get(f"W{mid[1:]}")
        print(f"  {mid}: born={m.birth_tick}, death={m.death_tick}, script={w.script}, reason={m.death_reason}")

    # We expect 3 manuscripts because:
    # 1. M1 born at tick 1 (Uncial)
    # 2. M2 born at tick 5 (Minuscule), kills M1. Alive count becomes 1.
    # 3. M3 born at tick 6 to meet demand=2 (Minuscule).
    assert len(state.registries.manuscripts) == 3
    m1 = state.registries.manuscripts.get("M1")

    w1 = state.registries.witnesses.get("W1")
    w2 = state.registries.witnesses.get("W2")
    w3 = state.registries.witnesses.get("W3")

    assert w1.script == Script.UNCIAL
    assert w2.script == Script.MINUSCULE
    assert w3.script == Script.MINUSCULE

    # M1 should have been killed at tick 5 when I2 (Minuscule) was born
    assert m1.death_tick == 5
    assert m1.death_reason == DeathReason.CULTURAL_REPLACEMENT
    assert m1.manuscript_id not in state.alive_manuscripts
