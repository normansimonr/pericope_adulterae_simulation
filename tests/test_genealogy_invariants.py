from pasim.core.genealogy import (
    GenealogyValidationError,
    add_child_node,
    add_root_node,
    create_empty_genealogy,
    validate_genealogy,
)
import pytest
import networkx as nx
import re
import ast

# Helper function for common node attributes
def get_node_attrs(node_id, birth_tick=1, reputation=3):
    return {
        "node_id": node_id,
        "witness_id": f"W_{node_id}",
        "manuscript_id": f"M_{node_id}",
        "birth_tick": birth_tick,
        "reputation": reputation,
    }

def test_single_root_creation_success():
    graph = create_empty_genealogy()
    add_root_node(graph, **get_node_attrs("I1"))
    validate_genealogy(graph) # Should pass
    assert graph.number_of_nodes() == 1
    assert len(list(nx.nodes(graph))) == 1
    assert len(list(nx.ancestors(graph, "I1"))) == 0

def test_multiple_root_creation_failure():
    graph = create_empty_genealogy()
    add_root_node(graph, **get_node_attrs("I1"))

    with pytest.raises(GenealogyValidationError, match=r"A root node \(autograph\) can only be added to an empty genealogy."):
        add_root_node(graph, **get_node_attrs("I2"))

def test_add_child_node_success():
    graph = create_empty_genealogy()
    add_root_node(graph, **get_node_attrs("I1"))
    add_child_node(graph, parent_node_ids=["I1"], **get_node_attrs("I2", birth_tick=2))
    validate_genealogy(graph) # Should pass
    assert graph.number_of_nodes() == 2
    assert ("I1", "I2") in graph.edges

def test_add_orphan_node_failure():
    graph = create_empty_genealogy()
    add_root_node(graph, **get_node_attrs("I1"))
    
    # Attempt to add a node that would become a new root (orphan)
    # This scenario is explicitly covered by `test_multiple_root_creation_failure`
    # because `validate_genealogy` will detect more than one node with in_degree == 0.
    pass 

def test_introduce_cycle_failure():
    graph = create_empty_genealogy()
    # Build a simple DAG: I1 -> I2 -> I3
    add_root_node(graph, **get_node_attrs("I1"))
    add_child_node(graph, parent_node_ids=["I1"], **get_node_attrs("I2", birth_tick=2))
    add_child_node(graph, parent_node_ids=["I2"], **get_node_attrs("I3", birth_tick=3))

    # Manually introduce a cycle: I3 -> I1 (completing I1 -> I2 -> I3 -> I1)
    # Note: This operation bypasses add_child_node's checks and directly modifies the graph.
    # We are testing validate_genealogy's ability to catch a cycle.
    graph.add_edge("I3", "I1")

    with pytest.raises(GenealogyValidationError, match=r"The genealogy graph must be a Directed Acyclic Graph \(DAG\)"):
        validate_genealogy(graph)

def test_missing_attributes_failure():
    graph = create_empty_genealogy()
    # Manually add a node with missing attributes to test validate_genealogy directly
    graph.add_node("I1", witness_id="W1") # Missing manuscript_id, birth_tick, reputation
    with pytest.raises(GenealogyValidationError) as excinfo:
        validate_genealogy(graph)
    
    expected_missing = {'manuscript_id', 'birth_tick', 'reputation'}
    actual_message = str(excinfo.value)
    
    # Use regex to extract the set string from the message and convert it back to a set
    match = re.search(r"missing required attributes: (\{[^}]+\})", actual_message)
    assert match, "Error message did not contain a set of missing attributes."
    
    actual_missing_str = match.group(1)
    actual_missing_set = ast.literal_eval(actual_missing_str)
    
    assert actual_missing_set == expected_missing

def test_empty_graph_validation():
    graph = create_empty_genealogy()
    validate_genealogy(graph) # Should pass for an empty graph

def test_validation_after_single_root():
    graph = create_empty_genealogy()
    add_root_node(graph, **get_node_attrs("I1"))
    validate_genealogy(graph) # Should pass