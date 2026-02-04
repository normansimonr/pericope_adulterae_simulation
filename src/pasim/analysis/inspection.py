"""
This module provides a set of pure, read-only helper functions for
inspecting the internal state of a `pasim` simulation run. These tools
are designed for developers to debug, verify, and understand the simulation's
output without modifying the core logic or state.
"""

from typing import Any, Dict, List, Tuple

import networkx as nx

from pasim.core.simulation_state import GenerationState

# A type alias for clarity
NodeID = Any


def manuscript_table(state: GenerationState) -> List[Dict[str, Any]]:
    """
    Creates a comprehensive table of all manuscripts in the simulation.

    This function joins data from the manuscript registry, the witness registry,
    and the genealogy graph to provide a unified view of each manuscript's
    properties and status.

    Args:
        state: The simulation's final GenerationState object.

    Returns:
        A list of dictionaries, where each dictionary represents a manuscript.
    """
    table = []
    for ms_id, manuscript in state.registries.manuscripts.items():
        instance_id = state.manuscript_to_instance_map.get(ms_id)
        is_alive = ms_id in state.alive_manuscripts

        script = None
        reputation = None

        if instance_id and instance_id in state.graph:
            node_data = state.graph.nodes[instance_id]
            witness_id = node_data.get("witness_id")
            if witness_id and witness_id in state.registries.witnesses:
                script = state.registries.witnesses.get(witness_id).script.value
            reputation = node_data.get("reputation")

        table.append({
            "manuscript_id": manuscript.manuscript_id,
            "birth_tick": manuscript.birth_tick,
            "death_tick": manuscript.death_tick,
            "region": manuscript.region.value,
            "location": manuscript.location,
            "material": manuscript.material.value,
            "script": script,
            "reputation": reputation,
            "alive": is_alive,
        })
    return table


def genealogy_edges(state: GenerationState) -> List[Tuple[NodeID, NodeID]]:
    """
    Returns a simple list of all parent-child edges in the genealogy graph.

    Args:
        state: The simulation's final GenerationState object.

    Returns:
        A list of tuples, where each tuple is `(parent_instance_id, child_instance_id)`.
    """
    return list(state.graph.edges)


def node_table(state: GenerationState) -> List[Dict[str, Any]]:
    """
    Creates a table of all witness instances (nodes) in the genealogy graph.

    This function provides a summary of each node's attributes and its
    connections within the graph.

    Args:
        state: The simulation's final GenerationState object.

    Returns:
        A list of dictionaries, where each dictionary represents a node.
    """
    table = []
    for node_id, data in state.graph.nodes(data=True):
        table.append({
            "instance_id": node_id,
            "manuscript_id": data.get("manuscript_id"),
            "witness_id": data.get("witness_id"),
            "birth_tick": data.get("birth_tick"),
            "reputation": data.get("reputation"),
            "parents": list(state.graph.predecessors(node_id)),
            "children": list(state.graph.successors(node_id)),
        })
    return table


def witness_text(state: GenerationState, witness_id: str) -> Any:
    """
    Retrieves the textual content (tagged string) for a given witness.

    NOTE: The current simulation engine does not yet store textual content.
    This function is a placeholder and will return None.

    Args:
        state: The simulation's final GenerationState object.
        witness_id: The ID of the witness to retrieve the text for.

    Returns:
        The tagged string (NumPy array) for the witness, or None if not found
        or not implemented.
    """
    # The current model does not yet store the tagged string for each witness.
    # When it does, this function will be updated to retrieve it.
    return None


def lineage_texts(state: GenerationState, leaf_instance_id: NodeID) -> List[Any]:
    """
    Traces the lineage of a leaf node back to its root and returns the text
    of each witness in chronological order.

    NOTE: Since witness_text is not yet fully implemented, this will return
    a list of None values.

    Args:
        state: The simulation's final GenerationState object.
        leaf_instance_id: The ID of the leaf node to trace back from.

    Returns:
        A list of tagged strings, ordered from the root to the specified leaf.
    """
    if leaf_instance_id not in state.graph:
        raise ValueError(f"Instance ID '{leaf_instance_id}' not found in graph.")

    # Find the ultimate ancestor (a root of the graph)
    ancestors = nx.ancestors(state.graph, leaf_instance_id)
    roots = {n for n in ancestors if state.graph.in_degree(n) == 0}
    # In a simple tree, there is one root. In a DAG, there could be more.
    # We find a path from one of the roots.
    if not roots:
        # The leaf_instance_id might be a root itself
        if state.graph.in_degree(leaf_instance_id) == 0:
            roots = {leaf_instance_id}
        else:
            raise ValueError("Could not find a root for the given leaf node.")

    root = list(roots)[0]

    # Find the path from the root to the leaf
    path = nx.shortest_path(state.graph, source=root, target=leaf_instance_id)

    texts = []
    for node_id in path:
        witness_id = state.graph.nodes[node_id].get("witness_id")
        if witness_id:
            texts.append(witness_text(state, witness_id))
    return texts


def to_networkx_copy(state: GenerationState) -> nx.DiGraph:
    """
    Returns a deep copy of the simulation's genealogy graph.

    This is useful for analysis or visualization tasks where you might want
    to modify the graph without affecting the original simulation state.

    Args:
        state: The simulation's final GenerationState object.

    Returns:
        A deep copy of the `networkx.DiGraph` object.
    """
    return state.graph.copy()
