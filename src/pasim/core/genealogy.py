"""
Defines the structural genealogy of witness instances in the simulation.

This module provides the primitives for creating, manipulating, and querying a
genealogy graph. The genealogy represents the structural and temporal
relationships between witness instances: who copied from whom and when. It is
modeled as a directed acyclic graph (DAG), where nodes are witness instances
and directed edges represent copying events (parent -> child).

Crucially, this layer is concerned only with the topology of the genealogy:
identities, ancestry, and timing. It does not store or have any knowledge of
the textual state (i.e., "tagged strings") of the witnesses. This separation
of concerns is a core design principle: it allows the same underlying
genealogical structure to be used as a basis for multiple, different models
of textual evolution. The textual state is managed and evolved in a separate
layer of the simulation, which then uses this genealogy as a fixed scaffold.

The functions in this module are deterministic and operate on an explicit
graph object (`networkx.DiGraph`). Node identity is explicit and stable,
ensuring that graph construction is reproducible.
"""

from typing import List, Optional, Any
import networkx as nx

# Define type hints for clarity
GenealogyGraph = nx.DiGraph
NodeID = str


def create_empty_genealogy() -> GenealogyGraph:
    """
    Creates an empty genealogy graph.

    The returned graph is a `networkx.DiGraph` instance, ready to be populated
    with nodes and edges representing the witness genealogy.

    Returns:
        An empty `networkx.DiGraph` object.
    """
    return nx.DiGraph()


def add_root_node(
    graph: GenealogyGraph,
    node_id: NodeID,
    witness_id: str,
    manuscript_id: str,
    birth_tick: int,
    reputation: int,
    death_tick: Optional[int] = None,
) -> None:
    """
    Adds a root witness instance to the genealogy (a node with no parents).

    This function is used to initialize a lineage with its first ancestor.
    It enforces the rule that a root node cannot have any predecessors in
    the graph.

    Invariants:
        - The `node_id` must be unique within the graph.
        - The node is added with no incoming edges.

    Args:
        graph: The genealogy graph to modify.
        node_id: The unique identifier for this witness instance node.
        witness_id: The foreign key to the Witness registry.
        manuscript_id: The foreign key to the Manuscript registry.
        birth_tick: The simulation tick at which this witness was created.
        reputation: The reputation score of this witness instance.
        death_tick: The simulation tick at which this witness ceased to be
                    available for copying (optional).

    Failure Conditions:
        - Raises `ValueError` if a node with `node_id` already exists.
    """
    if graph.has_node(node_id):
        raise ValueError(f"Node with ID '{node_id}' already exists in the graph.")

    graph.add_node(
        node_id,
        witness_id=witness_id,
        manuscript_id=manuscript_id,
        birth_tick=birth_tick,
        reputation=reputation,
        death_tick=death_tick,
    )


def add_child_node(
    graph: GenealogyGraph,
    node_id: NodeID,
    parent_node_ids: List[NodeID],
    witness_id: str,
    manuscript_id: str,
    birth_tick: int,
    reputation: int,
    death_tick: Optional[int] = None,
) -> None:
    """
    Adds a child witness instance descended from one or more parents.

    This represents a copying event, creating a new witness instance that is a
    descendant of the specified parent(s). It adds the new node and connects
    it to its parents with directed edges.

    Invariants:
        - The `node_id` must be unique.
        - Edges are added from each parent in `parent_node_ids` to the new child.

    Args:
        graph: The genealogy graph to modify.
        node_id: The unique identifier for the new child node.
        parent_node_ids: A list of node IDs for the parent(s).
        witness_id: The foreign key to the Witness registry.
        manuscript_id: The foreign key to the Manuscript registry.
        birth_tick: The simulation tick at which this witness was created.
        reputation: The reputation score of this witness instance.
        death_tick: The simulation tick at which this witness ceased to be
                    available for copying (optional).

    Failure Conditions:
        - Raises `ValueError` if a node with `node_id` already exists.
        - Raises `ValueError` if any parent ID in `parent_node_ids` does not
          exist in the graph.
        - Raises `networkx.HasACycle` if adding the node and its edges would
          introduce a cycle into the graph.
    """
    if not parent_node_ids:
        raise ValueError("Child node must have at least one parent.")

    if graph.has_node(node_id):
        raise ValueError(f"Node with ID '{node_id}' already exists in the graph.")

    for parent_id in parent_node_ids:
        if not graph.has_node(parent_id):
            raise ValueError(f"Parent node with ID '{parent_id}' does not exist.")

    # Temporarily add edges to check for cycles
    graph.add_node(node_id)
    edges = [(parent_id, node_id) for parent_id in parent_node_ids]
    graph.add_edges_from(edges)

    try:
        if not nx.is_directed_acyclic_graph(graph):
            raise nx.HasACycle("Adding this child node would create a cycle.")
    except nx.HasACycle:
        # Clean up the graph before re-raising
        graph.remove_node(node_id)
        raise

    # If no cycle, finalize the node's attributes
    nx.set_node_attributes(
        graph,
        {
            node_id: {
                "witness_id": witness_id,
                "manuscript_id": manuscript_id,
                "birth_tick": birth_tick,
                "reputation": reputation,
                "death_tick": death_tick,
            }
        },
    )


def validate_genealogy(graph: GenealogyGraph) -> None:
    """
    Performs structural sanity checks on the genealogy graph.

    This function verifies that the graph is a valid Directed Acyclic Graph (DAG)
    and that all nodes contain the minimally required attributes.

    Args:
        graph: The genealogy graph to validate.

    Failure Conditions:
        - Raises `TypeError` if the graph is not a `networkx.DiGraph`.
        - Raises `nx.NetworkXError` if the graph is not a DAG.
        - Raises `ValueError` if any node is missing required attributes.
    """
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("Graph must be a networkx.DiGraph instance.")

    if not nx.is_directed_acyclic_graph(graph):
        raise nx.NetworkXError(
            "The genealogy graph must be a Directed Acyclic Graph (DAG)."
        )

    required_attrs = {"witness_id", "manuscript_id", "birth_tick", "reputation"}
    for node_id, attrs in graph.nodes(data=True):
        missing_attrs = required_attrs - set(attrs.keys())
        if missing_attrs:
            raise ValueError(
                f"Node '{node_id}' is missing required attributes: {missing_attrs}"
            )


def get_parents(graph: GenealogyGraph, node_id: NodeID) -> List[NodeID]:
    """
    Gets the list of parents for a given node.

    Args:
        graph: The genealogy graph.
        node_id: The ID of the node whose parents are to be retrieved.

    Returns:
        A list of parent node IDs.
    """
    return list(graph.predecessors(node_id))


def get_children(graph: GenealogyGraph, node_id: NodeID) -> List[NodeID]:
    """
    Gets the list of children for a given node.

    Args:
        graph: The genealogy graph.
        node_id: The ID of the node whose children are to be retrieved.

    Returns:
        A list of child node IDs.
    """
    return list(graph.successors(node_id))


def get_roots(graph: GenealogyGraph) -> List[NodeID]:
    """
    Gets all root nodes in the graph (nodes with no parents).

    Args:
        graph: The genealogy graph.

    Returns:
        A list of root node IDs.
    """
    return [node for node, in_degree in graph.in_degree() if in_degree == 0]


def get_leaves(graph: GenealogyGraph) -> List[NodeID]:
    """
    Gets all leaf nodes in the graph (nodes with no children).

    Args:
        graph: The genealogy graph.

    Returns:
        A list of leaf node IDs.
    """
    return [node for node, out_degree in graph.out_degree() if out_degree == 0]
