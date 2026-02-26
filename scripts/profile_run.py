"""
This script serves as a lightweight, developer-focused entry point for
executing and inspecting a single simulation run. It uses the `run_single`
function to execute a predefined experiment configuration and prints a summary
of the results to the console.

Its purpose is to provide a quick way to verify that the end-to-end execution
pathway is working correctly after making changes to the simulation core or
configuration.
"""

import cProfile
import pstats

from pasim.analysis.inspection import (
    genealogy_edges,
    lineage_texts,
    manuscript_table,
    node_table,
)
from pasim.execution.runner import run_single


def profile_simulation():
    # Execute the baseline experiment with a fixed seed for reproducibility
    result = run_single("experiments/exp003_profiling/params.yaml", seed=42)

    # Print a summary of the results for quick inspection
    print("--- Simulation Run Summary ---")
    print(f"Final tick: {result.state.tick}")
    print(f"Alive manuscripts: {len(result.state.alive_manuscripts)}")
    print(f"Total manuscripts created: {len(result.state.registries.manuscripts)}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print("-" * 20)
    print("Telemetry (first 3 ticks):")
    for record in result.state.telemetry[:3]:
        print(f"  - Tick {record['tick']}: {record['alive_manuscripts']} alive / {record['total_manuscripts']} total")
    print("Telemetry (last 3 ticks):")
    for record in result.state.telemetry[-3:]:
        print(f"  - Tick {record['tick']}: {record['alive_manuscripts']} alive / {record['total_manuscripts']} total")
    print("--- End of Summary ---")

    print("\n--- Manuscripts ---")
    for row in manuscript_table(result.state):
        print(row)

    print("\n--- Genealogy Edges ---")
    print(genealogy_edges(result.state))

    print("\n--- Nodes ---")
    for row in node_table(result.state):
        print(row)

    # Pick a leaf node automatically to test lineage tracing
    leaf_nodes = [n for n in result.graph.nodes if result.graph.out_degree(n) == 0]
    if leaf_nodes:
        leaf = leaf_nodes[0]
        print(f"\n--- Lineage trace for leaf node: {leaf} ---")
        texts = lineage_texts(result.state, leaf)
        print(f"Lineage length: {len(texts)}")
        # When text is implemented, we can print the actual texts
        # print("Texts (root to leaf):", texts)
    else:
        print("\n--- No leaf nodes found for lineage trace. ---")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.run("profile_simulation()")
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(30)
    stats.dump_stats("profile_stats.prof")
