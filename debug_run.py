import numpy as np

from pasim.execution.runner import run_single


def debug_run():
    params_path = "experiments/exp001_baseline/params.yaml"
    print(f"Running single run of {params_path}...")

    # Run with a fixed seed for reproducibility
    seed = 42
    result = run_single(params_path, seed=seed, persistence_level="minimal")

    print(f"Run ID: {result.run_id}")
    print(f"Seed: {result.seed}")

    for regime, replay in result.replays.items():
        print(f"\n--- Regime: {regime} ---")
        # Find some interesting nodes
        nodes = sorted(result.genealogy_snapshot.nodes, key=lambda n: n.birth_tick)

        # Autograph
        root = nodes[0]
        root_text = replay.instance_texts[root.instance_id]
        print(f"Autograph ({root.instance_id}, tick {root.birth_tick}): {root_text[:20]}...")

        # Innovator
        innovator_id = replay.innovator_id
        innovator_node = next(n for n in nodes if n.instance_id == innovator_id)
        innovator_text = replay.instance_texts[innovator_id]
        print(f"Innovator ({innovator_id}, tick {innovator_node.birth_tick}): {innovator_text[:20]}...")

        # Some later nodes
        later_nodes = [n for n in nodes if n.birth_tick > 1000][:3]
        for n in later_nodes:
            text = replay.instance_texts[n.instance_id]
            print(f"Node {n.instance_id} (tick {n.birth_tick}): {text[:20]}...")

        # Majority text
        majority_text = np.array(replay.majority_text_segments)
        survivor_ids = result.survivor_sampling_result.sampled_witness_ids
        print(f"Survivors used for Majority Text: {len(survivor_ids)}")
        print(f"Majority Text (first 20): {majority_text[:20]}...")
        print(f"Majority Text (sum): {np.sum(majority_text)}")


if __name__ == "__main__":
    debug_run()
