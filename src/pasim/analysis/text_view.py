"""
This module provides a text-based visualization utility for inspecting
the differences between parent and child texts in a simulation.
"""

import numpy as np

from pasim.core.simulation_state import GenerationState


def print_textual_diffs(state: GenerationState):
    """
    Prints a textual diff for each parent-child relationship in the genealogy.

    This is a debug utility for inspecting mutations. It prints to stdout.
    """
    print("\n--- Textual Lineage Diffs ---")

    edges = sorted(
        state.graph.edges(data=True),
        key=lambda edge: state.graph.nodes[edge[1]]["birth_tick"],
    )

    if not edges:
        print("No parent-child edges found in the graph.")
        return

    for parent_id, child_id, _ in edges:
        parent_text = state.registries.instance_texts.get(parent_id)
        child_text = state.registries.instance_texts.get(child_id)

        if parent_text is None or child_text is None:
            continue

        _print_single_diff(parent_id, child_id, parent_text, child_text)


def _print_single_diff(
    parent_id: str,
    child_id: str,
    parent_text: np.ndarray,
    child_text: np.ndarray,
    truncate: int = 120,
):
    """Prints a single parent-child diff."""
    print(f"\nParent {parent_id} -> Child {child_id}")

    if len(parent_text) > truncate:
        _print_truncated_diff(parent_text, child_text, truncate)
    else:
        parent_str = " ".join(map(str, parent_text))
        child_str = " ".join(map(str, child_text))

        diff_markers = np.where(parent_text != child_text)[0]
        diff_line = ""
        last_marker = -1
        for marker in diff_markers:
            # Calculate the position of the marker in the string
            # This assumes single-digit numbers and single spaces
            pos = marker * 2
            diff_line += " " * (pos - last_marker - 1) + "^"
            last_marker = pos

        print(f"Parent:  {parent_str}")
        print(f"Child :  {child_str}")
        if diff_line:
            print(f"Diff  :  {diff_line}")


def _print_truncated_diff(parent_text: np.ndarray, child_text: np.ndarray, truncate: int):
    """Prints a diff for truncated long texts."""
    half = truncate // 2
    parent_head = " ".join(map(str, parent_text[:half]))
    parent_tail = " ".join(map(str, parent_text[-half:]))
    child_head = " ".join(map(str, child_text[:half]))
    child_tail = " ".join(map(str, child_text[-half:]))

    print(f"Parent:  {parent_head} ... {parent_tail}")
    print(f"Child :  {child_head} ... {child_tail}")

    diff_markers = np.where(parent_text != child_text)[0]
    head_diff_markers = diff_markers[diff_markers < half]
    tail_diff_markers = diff_markers[diff_markers >= len(parent_text) - half]

    diff_line = ""
    if len(head_diff_markers) > 0:
        last_marker = -1
        for marker in head_diff_markers:
            pos = marker * 2
            diff_line += " " * (pos - last_marker - 1) + "^"
            last_marker = pos

    diff_line += " " * (len(parent_head) - len(diff_line)) + " ... "

    if len(tail_diff_markers) > 0:
        last_marker = -1
        for marker in tail_diff_markers:
            pos = (marker - (len(parent_text) - half)) * 2
            diff_line += " " * (pos - last_marker - 1) + "^"
            last_marker = pos

    if len(head_diff_markers) > 0 or len(tail_diff_markers) > 0:
        print(f"Diff  :  {diff_line}")
    else:
        print("(No differences in head/tail, output truncated)")
