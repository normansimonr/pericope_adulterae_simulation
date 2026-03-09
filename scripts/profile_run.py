import cProfile
import pstats
import time

from pasim.execution.runner import run_single


def profile_simulation():
    print("Starting high-demand simulation...")
    start_time = time.time()
    result = run_single("experiments/exp004_high_demand/params.yaml", seed=42, persistence_level="minimal")
    end_time = time.time()

    print("--- Simulation Run Summary ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Final tick: {result.state.tick}")
    print(f"Alive manuscripts: {len(result.state.alive_manuscripts)}")
    print(f"Total manuscripts created: {len(result.state.registries.manuscripts)}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print("--- End of Summary ---")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.run("profile_simulation()")
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(50)
    stats.dump_stats("profile_stats.prof")
