"""
This script serves as a lightweight, developer-focused entry point for
executing and inspecting a single simulation run. It uses the `run_single`
function to execute a predefined experiment configuration and prints a summary
of the results to the console.

Its purpose is to provide a quick way to verify that the end-to-end execution
pathway is working correctly after making changes to the simulation core or
configuration.
"""
from pasim.execution.runner import run_single

# Execute the baseline experiment with a fixed seed for reproducibility
result = run_single("experiments/exp001_baseline/params.yaml", seed=42)

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
