import numpy as np
import pytest
from pasim.config.schema import SimulationConfig
from pasim.core.mutation import mutate_tagged_string
from pasim.core.scribal_rules import apply_scribal_rule

@pytest.fixture
def config():
    return SimulationConfig(
        total_ticks=10,
        text_length=10,
        demand_schedule={0: 1},
        reputation_distribution={1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        pa_regime="insertion",
        pa_intervention_year=5,
        pa_intervention_region="Asia Minor",
        pa_innovator_reputation=5.0,
    )

def test_mutation_of_zero_single_parent(config):
    rng = np.random.default_rng(42)
    # Reputation 1 usually has some mutation rate
    # Let's check what the default mapping is or provide one
    mutation_mapping = {1: 0.5} 
    
    # Single parent with some zeros
    parent_text = np.array([0, 1, 0, 2, 0, 3, 0, 4, 0, 5], dtype=np.int16)
    exemplar_texts = [parent_text]
    
    # Current behavior (before fix): 0s can mutate
    # We want to check if they DO mutate currently
    
    # To be sure we see mutations, let's try many times or use high rate
    child_text = apply_scribal_rule(
        exemplar_texts, rng, reputation=1, config=config, mutation_mapping=mutation_mapping
    )
    
    # If any position that was 0 is now not 0, it means it mutated.
    zeros_mask = (parent_text == 0)
    mutated_zeros = child_text[zeros_mask]
    
    print(f"Parent: {parent_text}")
    print(f"Child:  {child_text}")
    
    # BEFORE FIX: This might fail (i.e., some zeros might have mutated)
    # The new rule says they MUST NOT mutate.
    assert np.all(child_text[zeros_mask] == 0), f"Zeros mutated in single parent case! {child_text[zeros_mask]}"

def test_mutation_of_zero_multi_parent(config):
    rng = np.random.default_rng(42)
    mutation_mapping = {1: 0.9} # High rate to ensure we see it
    
    # Multi parent: existing logic remains unchanged (0s CAN mutate)
    parent1 = np.zeros(10, dtype=np.int16)
    parent2 = np.zeros(10, dtype=np.int16)
    exemplar_texts = [parent1, parent2]
    
    child_text = apply_scribal_rule(
        exemplar_texts, rng, reputation=1, config=config, mutation_mapping=mutation_mapping
    )
    
    # In multi-parent case, it is STILL POSSIBLE to mutate 0 to something else
    # (unless the existing logic already prevented it, but it shouldn't)
    print(f"Multi-parent Child: {child_text}")
    # At least some should have mutated if rate is 0.9 and length is 10
    assert not np.all(child_text == 0), "Multi-parent 0s did not mutate (they should be allowed to)"
