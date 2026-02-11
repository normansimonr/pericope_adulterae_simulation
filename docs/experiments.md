# Running Experiments in `pasim`

This document provides a guide to setting up and running simulation experiments within the `pasim` framework. It covers the directory structure, how to configure and execute single and parallel Monte Carlo runs, and explains the structure of the experiment parameter files.

## A. Experiment Directory Structure

All experiment definitions and their resulting data are organized within the `experiments/` directory in the project root.

```
experiments/
  exp001_baseline/          # Each experiment resides in its own dedicated directory.
    params.yaml             # This file contains all configuration for the experiment.
    runs/                   # This directory is auto-created by the runner.
      1/                    # Each sub-directory represents a single simulation run.
        config.yaml         # Exact configuration used for this specific run.
        run_metadata.json   # Metadata about the run (seed, final tick, etc.).
        genealogy.json      # The full genealogy graph.
        instances.json      # Detailed metadata for all witness instances.
        manuscripts.json    # Full details of all manuscripts.
        instance_texts.tsv  # Textual content of all witness instances.
        telemetry.json      # Per-tick metrics.
        events.log          # Chronological log of key simulation events.
      2/                    # Another run directory.
      ...
  exp002_sweep/
    params.yaml
    runs/
      ...
```

**Key Points:**

*   **Dedicated Directories:** Each unique experiment (e.g., `exp001_baseline`, `exp002_sweep`) must have its own subdirectory directly under `experiments/`.
*   **Single `params.yaml`:** Every experiment directory must contain exactly one `params.yaml` file. This file specifies both the overall experiment orchestration (like the number of runs) and the detailed simulation configuration.
*   **Auto-generated `runs/`:** The `runs/` subdirectory is automatically created by the experiment runner when the experiment is executed. It houses the results of individual simulation runs.
*   **Indexed Run Folders:** Individual run folders within `runs/` are named by integer counters (e.g., `1/`, `2/`, `3/`). These are simple incremental indices, not timestamps or zero-padded numbers.

## B. Minimal Single-Run Setup

To set up and execute a single simulation run:

1.  **Create an Experiment Directory:**
    Choose a descriptive name for your experiment (e.g., `exp001_test`). Create a new directory for it:
    ```bash
    mkdir experiments/exp001_test
    ```

2.  **Create `params.yaml`:**
    Copy the canonical template (`experiments/params_template.yaml`) into your new experiment directory and rename it to `params.yaml`:
    ```bash
    cp experiments/params_template.yaml experiments/exp001_test/params.yaml
    ```
    Open `experiments/exp001_test/params.yaml` with a text editor.

3.  **Configure for a Single Run:**
    Ensure the `n_runs` parameter in `experiments/exp001_test/params.yaml` is set to `1`:
    ```yaml
    # experiments/exp001_test/params.yaml
    n_runs: 1
    # ... other parameters ...
    ```

4.  **Trigger Execution:**
    The `pasim` framework uses `pasim.execution.orchestrator.run_experiment` as the entry point for executing experiments. You can run your experiment programmatically (e.g., from an IPython console or a Python script):

    ```python
    from pasim.execution.orchestrator import run_experiment

    # The path should be relative to the project root.
    experiment_summary = run_experiment("experiments/exp001_test/params.yaml")
    print(experiment_summary)
    ```
    Upon successful execution, a `runs/` directory will be created inside `experiments/exp001_test/`, containing `1/` with all the persisted results for your single run.

## C. Params File Structure

The `params.yaml` file for each experiment serves a dual purpose: it configures the overall experiment orchestration (e.g., how many runs, retry logic) and defines the detailed parameters for the `pasim` simulation itself.

### Top-level Experiment Orchestration Parameters

These fields control how the `pasim` framework manages the execution of your experiment.

*   **`n_runs`** (int, required)
    *   **Description:** The number of independent Monte Carlo simulation runs to execute for this experiment.
    *   **Influence:**
        *   **Parallel Execution:** Determines how many parallel processes (up to `num_processes` if specified, or CPU count) will be spawned to execute individual simulation runs.
        *   **Persistence:** A separate `runs/` subdirectory will be created for each of the `n_runs` runs.
    *   **Default Behavior:** Must be explicitly set.
    *   **Example:** `n_runs: 100` for 100 Monte Carlo runs.

*   **`max_retries`** (int, required)
    *   **Description:** The maximum number of times the framework will attempt to re-run an individual simulation if it fails.
    *   **Influence:** Affects the robustness of parallel execution. A higher number increases the chance of completing all `n_runs` successfully even if transient errors occur.
    *   **Default Behavior:** Must be explicitly set. A common value is `3`.
    *   **Example:** `max_retries: 3`

*   **`seed`** (int or null, optional)
    *   **Description:** The master seed for the random number generators used across the entire experiment.
    *   **Influence:**
        *   **Reproducibility:** If provided, ensures that the *entire batch* of `n_runs` will produce identical results if run again with the same seed.
        *   **Reproducibility:** Each individual run will derive a unique, deterministic seed from this master seed, guaranteeing independent but reproducible stochastic behavior for each run.
    *   **Default Behavior:** If `null` or omitted, a default fixed seed (e.g., `20240105`) is used by the framework to maintain reproducibility of the batch.
    *   **Example:** `seed: 12345` or `seed: null`

### Simulation Configuration Fields

These fields define the intrinsic dynamics and historical context of a single `pasim` simulation run. They directly map to the `pasim.config.schema.SimulationConfig` Pydantic model. Refer to `src/pasim/config/schema.py` for the most authoritative and up-to-date schema definition, including all validation rules.

#### Core Simulation Dynamics

These parameters control the fundamental mechanical processes within the simulation.

*   **`total_ticks`** (int, required)
    *   **Description:** The total duration of the simulation, measured in discrete time steps (ticks).
    *   **Constraints:** Must be an integer greater than or equal to 1.
    *   **Influence:** Dictates the simulation's length and thus the potential for historical events and textual evolution.
    *   **Example:** `total_ticks: 100`

*   **`text_length`** (int, optional)
    *   **Description:** The number of segments (tokens) in the "tagged string" that represents the textual content of a witness.
    *   **Constraints:** Must be an integer greater than or equal to 1.
    *   **Default:** `200`
    *   **Influence:** Affects the granularity and complexity of textual mutation and analysis.
    *   **Example:** `text_length: 500`

*   **`p_region_migration`** (float, optional)
    *   **Description:** The probability (per tick) that any given manuscript will migrate from its current region to a different, randomly selected region.
    *   **Constraints:** Must be a float between `0.0` and `1.0` (inclusive).
    *   **Default:** `0.0` (no regional migration)
    *   **Influence:** Impacts the spatial distribution of manuscripts and thus the available exemplars for copying in different regions.
    *   **Example:** `p_region_migration: 0.01` (1% chance of migrating per tick)

*   **`p_internal_relocation`** (float, optional)
    *   **Description:** The probability (per tick) that any given manuscript will relocate its precise (x,y) coordinates *within* its current region.
    *   **Constraints:** Must be a float between `0.0` and `1.0` (inclusive).
    *   **Default:** `0.0` (no internal relocation)
    *   **Influence:** Affects the spatial proximity of manuscripts within a region, which can influence exemplar selection.
    *   **Example:** `p_internal_relocation: 0.05` (5% chance of relocating within region per tick)

*   **`reputation_distribution`** (Dict[int, float], required)
    *   **Description:** Defines the probability distribution from which the reputation score (an integer from 1 to 5) for *newly created* witness instances is sampled.
    *   **Constraints:**
        *   Keys must be integers `1`, `2`, `3`, `4`, `5`.
        *   Values must be floats representing probabilities (non-negative).
        *   The sum of all probabilities must equal `1.0`.
    *   **Influence:** Directly impacts the "error intensity" (expected mutation proportion) of newly copied texts, as higher reputation typically correlates with lower error rates.
    *   **Example:**
        ```yaml
        reputation_distribution:
          1: 0.1 # 10% chance of reputation 1 (lowest)
          2: 0.2
          3: 0.4 # 40% chance of reputation 3 (median)
          4: 0.2
          5: 0.1 # 10% chance of reputation 5 (highest)
        ```

*   **`death_ticks`** (List[int], required)
    *   **Description:** A pre-generated list of integer ticks. When a new manuscript is created, its `death_tick` is drawn sequentially from this list.
    *   **Constraints:** Each element must be a non-negative integer. The list should be sufficiently long to accommodate all expected manuscript creations.
    *   **Influence:** Determines the lifespan of individual manuscripts, affecting the overall population dynamics and the availability of exemplars over time.
    *   **Example:** `death_ticks: [20, 30, 40, 50, 60]`

#### Historical Events (Shocks)

These parameters define discrete, time-bound events that act as "shocks" to the simulation, potentially affecting existing manuscripts.

*   **`persecutions`** (List[PersecutionEventConfig], optional)
    *   **Description:** A list of `PersecutionEventConfig` objects, each defining a historical persecution event. A persecution destroys a specified proportion of eligible manuscripts in defined regions during a given time window.
    *   **Constraints for each `PersecutionEventConfig`:**
        *   `start_tick` (int): The simulation tick when the persecution begins. Must be `>= 0`.
        *   `end_tick` (int, optional): The tick when the persecution ends. If omitted, the event is instantaneous at `start_tick`. Must be `>= 0` and `>= start_tick`.
        *   `regions` (List[str]): A list of region names (e.g., "Asia Minor", "Egypt") where the persecution applies. Must be valid `Region` enum values.
        *   `kill_proportion` (float): The proportion of eligible manuscripts (in specified regions, active during the event) that will be destroyed. Must be between `0.0` and `1.0` (inclusive).
    *   **Influence:** Can drastically reduce manuscript populations, impacting the availability of exemplars and potentially accelerating textual divergence.
    *   **Example:**
        ```yaml
        persecutions:
          - start_tick: 50
            end_tick: 55
            regions:
              - "Asia Minor"
              - "Egypt"
            kill_proportion: 0.5 # 50% of eligible manuscripts in these regions destroyed between tick 50 and 55
          - start_tick: 80
            regions:
              - "Levant"
            kill_proportion: 0.8 # Instantaneous 80% destruction in Levant at tick 80
        ```

#### Environmental Regimes (Transitions)

These parameters define long-term, gradual shifts in the simulation's environment, influencing the properties of *newly created* entities (manuscripts or witness instances).

*   **`material_transitions`** (List[MaterialTransitionConfig], optional)
    *   **Description:** A schedule of `MaterialTransitionConfig` objects, defining how the probability distribution for the material of *newly created* manuscripts changes over time.
    *   **Constraints for each `MaterialTransitionConfig`:**
        *   `start_tick` (int): The tick at which this distribution becomes active. `start_tick` values across the list must be strictly increasing. Must be `>= 0`.
        *   `distribution` (Dict[str, float]): A dictionary mapping material names (e.g., "parchment", "papyrus", "paper") to their probabilities. Must be valid `Material` enum values. Probabilities must be non-negative and sum to `1.0`.
    *   **Influence:** Models technological shifts in manuscript production, potentially affecting preservation and copying characteristics.
    *   **Example:**
        ```yaml
        material_transitions:
          - start_tick: 0
            distribution:
              parchment: 0.8
              papyrus: 0.2
              paper: 0.0
          - start_tick: 50 # From tick 50 onwards, a new distribution is active
            distribution:
              parchment: 0.6
              papyrus: 0.1
              paper: 0.3
        ```

*   **`script_transitions`** (List[ScriptTransitionConfig], optional)
    *   **Description:** A schedule of `ScriptTransitionConfig` objects, defining how the probability distribution for the script style of *newly created* witness instances changes over time.
    *   **Constraints for each `ScriptTransitionConfig`:**
        *   `start_tick` (int): The tick at which this distribution becomes active. `start_tick` values across the list must be strictly increasing. Must be `>= 0`.
        *   `distribution` (Dict[str, float]): A dictionary mapping script names (e.g., "uncial", "minuscule") to their probabilities. Must be valid `Script` enum values. Probabilities must be non-negative and sum to `1.0`.
    *   **Influence:** Models cultural and stylistic shifts in writing, potentially impacting readability and scribal error types (though the latter is not directly modeled by this transition).
    *   **Example:**
        ```yaml
        script_transitions:
          - start_tick: 0
            distribution:
              uncial: 1.0
              minuscule: 0.0
          - start_tick: 70 # From tick 70 onwards, minuscule script becomes more common
            distribution:
              uncial: 0.3
              minuscule: 0.7
        ```

#### Structural Drivers

These parameters directly input the simulation's fundamental mechanics, such as the demand for new manuscripts.

*   **`demand_schedule`** (Dict[int, Dict[str, int]], required)
    *   **Description:** Defines the target number of alive manuscripts for specific regions at particular simulation ticks. The simulation will attempt to spawn new manuscripts to meet these minimum numbers.
    *   **Constraints:**
        *   Top-level keys are integers representing `tick` values. Must be `>= 0`.
        *   Nested keys are string names of regions (e.g., "Asia Minor", "Egypt"). Must be valid `Region` enum values.
        *   Nested values are integers representing the `demanded_count` for that region at that tick. Must be `>= 0`.
    *   **Default Behavior:** If a `tick` is not explicitly defined in the schedule, the simulation will use the `demanded_count` from the *last known tick* (the highest `start_tick` less than or equal to the current tick). This ensures continuity of demand.
    *   **Influence:** Drives the creation of new manuscripts, which is fundamental to the growth of the genealogy graph and the propagation of textual variations.
    *   **Example:**
        ```yaml
        demand_schedule:
          0: # Demand at tick 0
            Asia Minor: 1 # Ensure at least 1 manuscript in "Asia Minor" at tick 0
            Egypt: 0
            Levant: 0
          10: # Demand at tick 10
            Asia Minor: 2
            Egypt: 1
          20: # Demand at tick 20 (and all subsequent ticks until a new entry)
            Asia Minor: 3
            Egypt: 2
            Levant: 1
        ```
