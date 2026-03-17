import logging

from pasim.execution.runner import run_single

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)


def test_innovation_survival():
    params_path = "experiments/exp001_baseline/params.yaml"
    # Run with a few different seeds to see if it's consistent
    for seed in [42, 123, 999]:
        print(f"\n--- Testing Seed {seed} ---")
        result = run_single(params_path, seed=seed, persistence_level="full")

        for regime, replay in result.replays.items():
            total = len(result.genealogy_snapshot.nodes)
            pct = replay.pct_all_witnesses_with_pa
            # In insertion, pct is fraction with PA (the innovation)
            # In omission, pct is fraction with PA (the autograph tradition)

            if regime == "insertion":
                count_with_innovation = round(pct * total)
                print(f"Regime {regime}: Innovation (PA) count: {count_with_innovation} / {total} ({pct * 100:.4f}%)")
            else:
                count_with_innovation = round((1 - pct) * total)
                print(f"Regime {regime}: Innovation (Omission) count: {count_with_innovation} / {total} ({(1 - pct) * 100:.4f}%)")


if __name__ == "__main__":
    test_innovation_survival()
