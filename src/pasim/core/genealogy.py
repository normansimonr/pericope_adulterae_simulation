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

from typing import List, TypeAlias

import networkx as nx

# Define type hints for clarity
NodeID = str
GenealogyGraph: TypeAlias = nx.DiGraph


class GenealogyValidationError(Exception):
    """Custom exception for genealogy validation errors."""

    pass


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
) -> None:
    """
    Adds a root witness instance to the genealogy (a node with no parents).

    This function is used to initialize a lineage with its first ancestor.
    It enforces the rule that a root node cannot have any predecessors in
    the graph.

    Invariants:
        - The `node_id` must be unique within the graph.
        - The node is added with no incoming edges.
        - A root can only be added to an empty graph, ensuring only one autograph.

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
        - Raises `GenealogyValidationError` if the graph is not empty.
        - Raises `ValueError` if a node with `node_id` already exists.
    """
    if graph.number_of_nodes() != 0:
        raise GenealogyValidationError("A root node (autograph) can only be added to an empty genealogy.")

    if graph.has_node(node_id):
        raise ValueError(f"Node with ID '{node_id}' already exists in the graph.")

    graph.add_node(
        node_id,
        witness_id=witness_id,
        manuscript_id=manuscript_id,
        birth_tick=birth_tick,
        reputation=reputation,
    )
    validate_genealogy(graph)


def add_child_node(
    graph: GenealogyGraph,
    node_id: NodeID,
    parent_node_ids: List[NodeID],
    witness_id: str,
    manuscript_id: str,
    birth_tick: int,
    reputation: int,
) -> None:
    """
    Adds a child witness instance descended from one or more parents.

    This represents a copying event, creating a new witness instance that is a
    descendant of the specified parent(s). It adds the new node and connects
    it to its parents with directed edges. The genealogy's invariants are
    validated after the operation.

    Invariants:
        - The `node_id` must be unique.
        - Edges are added from each parent in `parent_node_ids` to the new child.
        - The resulting graph must remain a valid genealogy.

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
        - Raises `ValueError` if a node with `node_id` already exists, if any
          parent ID does not exist, or if no parents are provided.
        - Raises `GenealogyValidationError` if adding the node violates any
          genealogy invariants (e.g., introduces a cycle).
    """
    if not parent_node_ids:
        raise ValueError("Child node must have at least one parent.")

    if graph.has_node(node_id):
        raise ValueError(f"Node with ID '{node_id}' already exists in the graph.")

    for parent_id in parent_node_ids:
        if not graph.has_node(parent_id):
            raise ValueError(f"Parent node with ID '{parent_id}' does not exist.")

    # Add the node and edges, then validate.
    graph.add_node(
        node_id,
        witness_id=witness_id,
        manuscript_id=manuscript_id,
        birth_tick=birth_tick,
        reputation=reputation,
    )
    edges = [(parent_id, node_id) for parent_id in parent_node_ids]
    graph.add_edges_from(edges)

    try:
        validate_genealogy(graph)
    except GenealogyValidationError as e:
        # If validation fails, revert the changes to maintain a valid state.
        graph.remove_node(node_id)
        raise GenealogyValidationError(f"Failed to add child '{node_id}': {e}") from e


def validate_genealogy(graph: GenealogyGraph) -> None:
    """
    Performs structural sanity checks on the genealogy graph.

    This function is the single authoritative validator for genealogy correctness.
    It enforces that the graph is a valid Directed Acyclic Graph (DAG) with at
    most one root, and that all nodes are well-formed.

    Invariants Checked:
        1. The graph must be a `networkx.DiGraph`.
        2. The graph must be acyclic (a DAG).
        3. There can be at most one root (autograph). Any node with an
           in-degree of zero is considered a root. This implicitly prevents
           orphaned nodes or disconnected components.
        4. All nodes must have a standard set of attributes.

    Args:
        graph: The genealogy graph to validate.

    Failure Conditions:
        - Raises `TypeError` if the graph is not a `networkx.DiGraph`.
        - Raises `GenealogyValidationError` if any invariant is violated.
    """
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("Graph must be a networkx.DiGraph instance.")

    # Invariant 2: The graph must be acyclic.
    if not nx.is_directed_acyclic_graph(graph):
        raise GenealogyValidationError("The genealogy graph must be a Directed Acyclic Graph (DAG).")

    # Invariant 3: At most one root. This also prevents orphan instances,
    # as any orphan would be a root of its own disconnected component.
    roots = get_roots(graph)
    if len(roots) > 1:
        raise GenealogyValidationError(f"Genealogy cannot have more than one root (autograph). " f"Found {len(roots)} roots: {roots}")

    # Invariant 4: All nodes must have required attributes.
    required_attrs = {"witness_id", "manuscript_id", "birth_tick", "reputation"}
    for node_id, attrs in graph.nodes(data=True):
        missing_attrs = required_attrs - set(attrs.keys())
        if missing_attrs:
            raise GenealogyValidationError(f"Node '{node_id}' is missing required attributes: {missing_attrs}")


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
