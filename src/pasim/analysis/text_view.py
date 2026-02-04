"""
This module provides a text-based visualization utility for inspecting
the differences between parent and child texts in a simulation.
"""

import numpy as np

from pasim.core.simulation_state import GenerationState

# ANSI escape codes for colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"


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
            # This can happen if a node is an initial root and has no parent.
            # Or if text was not stored for some reason (shouldn't happen here).
            continue

        # Ensure texts are of the same length for direct comparison
        if len(parent_text) != len(child_text):
            print(f"Warning: Parent {parent_id} and Child {child_id} texts have different lengths.")
            print(f"Parent ({parent_id}): {parent_text}")
            print(f"Child ({child_id}): {child_text}")
            continue

        _print_single_diff(parent_id, child_id, parent_text, child_text)


def _format_diff_line(
    original_text: np.ndarray,
    compared_text: np.ndarray,
    color_on_diff: str,
    color_reset: str,
    indices_to_compare: np.ndarray,
) -> str:
    """Helper to format a line with colored differences."""
    formatted_tokens = []
    for i in indices_to_compare:
        if original_text[i] != compared_text[i]:
            formatted_tokens.append(f"{color_on_diff}{original_text[i]}{color_reset}")
        else:
            formatted_tokens.append(str(original_text[i]))
    return " ".join(formatted_tokens)


def _print_single_diff(
    parent_id: str,
    child_id: str,
    parent_text: np.ndarray,
    child_text: np.ndarray,
    truncate: int = 120,
):
    """Prints a single parent-child diff in a git-like inline style with colors."""
    print(f"\nParent {parent_id} -> Child {child_id}")

    text_length = len(parent_text)  # Assumed same length as per context.

    if text_length > truncate:
        _print_truncated_diff_colored(parent_id, child_id, parent_text, child_text, truncate)
    else:
        # Full text display
        indices = np.arange(text_length)
        parent_formatted = _format_diff_line(parent_text, child_text, COLOR_RED, COLOR_RESET, indices)
        child_formatted = _format_diff_line(child_text, parent_text, COLOR_GREEN, COLOR_RESET, indices)

        print(f"Parent:  {parent_formatted}")
        print(f"Child :  {child_formatted}")


def _print_truncated_diff_colored(
    parent_id: str,
    child_id: str,
    parent_text: np.ndarray,
    child_text: np.ndarray,
    truncate: int,
):
    """Prints a truncated git-like inline diff for long texts."""
    half = truncate // 2
    text_length = len(parent_text)

    # Process head
    head_indices = np.arange(half)
    parent_head_formatted = _format_diff_line(parent_text, child_text, COLOR_RED, COLOR_RESET, head_indices)
    child_head_formatted = _format_diff_line(child_text, parent_text, COLOR_GREEN, COLOR_RESET, head_indices)

    # Process tail
    tail_indices = np.arange(text_length - half, text_length)
    parent_tail_formatted = _format_diff_line(parent_text, child_text, COLOR_RED, COLOR_RESET, tail_indices)
    child_tail_formatted = _format_diff_line(child_text, parent_text, COLOR_GREEN, COLOR_RESET, tail_indices)

    print(f"Parent:  {parent_head_formatted} ... {parent_tail_formatted}")
    print(f"Child :  {child_head_formatted} ... {child_tail_formatted}")
    print(f"(Output truncated, showing head and tail. Full length: {text_length})")
