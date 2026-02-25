# Project Overview: Pericope Adulterae Simulation (pasim)

This document provides an overview of the `pasim` project, outlining its purpose, architectural structure, and general functioning. This context is read by GEMINI CLI at each session start.

## Purpose

The primary goal of this project is to simulate the textual transmission of the New Testament passage known as the Pericope Adulterae (John 7:53-8:11). It provides a framework for researchers to conduct computational simulations, specifically Monte Carlo simulations, of the copying process of this Greek passage over centuries. This allows for the study of how textual variations might have emerged and propagated through historical manuscript traditions.

## Core Concepts

To accurately model textual transmission, the simulation distinguishes between several key entities:

-   **Manuscript**: Represents a physical object (e.g., a codex, scroll) with tangible attributes such as its `birth_tick`, `death_tick`, `material` (parchment, papyrus, paper), `region`, and `location`. These attributes belong exclusively to the physical manuscript. Manuscripts are tracked in a `ManuscriptRegistry`.

-   **Witness**: Represents the textual content found within a specific manuscript. A Manuscript owns exactly one Witness. A Witness has its own unique ID and is linked to its parent Manuscript. Witnesses are tracked in a `WitnessRegistry`.

-   **Witness Instance / Genealogy Node**: This is an abstract representation in the genealogy graph. Each Witness has exactly one Witness Instance, which corresponds to a node in the `networkx` graph. This node stores only structural information (like its ID and links to its Witness and Manuscript via foreign keys) and its `birth_tick`. Crucially, **it does NOT store regional, geographic, material, or `death_tick` data**, as these belong to the `Manuscript` object. The graph thus focuses purely on the lineage and copying relationships between textual instances.

    To ensure efficient lookup between a `Manuscript` (with its rich metadata) and its corresponding `Witness Instance` (graph node), a direct mapping (`manuscript_to_instance_map`) is maintained within the simulation's `GenerationState` (now housed in `core/simulation_state.py`). This map is populated at the exact moment a `Manuscript` and its `Witness Instance` are created together in an atomic operation (specifically within `_spawn_new_manuscripts_from_demand`). This guarantees the correctness of the mapping and allows for O(1) retrieval of a `Witness Instance`'s ID given a `Manuscript`'s ID, without needing to iterate through intermediate registries or graph nodes.

The relationship can be visualized as:
`Manuscript` (physical attributes)
  `|`
  `+-- owns --> Witness` (textual content)
           `|`
           `+-- represented by --> Witness Instance` (graph node)

This separation ensures that the abstract genealogical structure remains clean and unburdened by the evolving physical properties of the manuscripts.

## Configuration and Parameterization

The simulation is designed to be fully parameter-driven, allowing researchers to define and explore a wide range of historical scenarios without altering the core source code. All configuration is managed through a centralized, validated schema.

-   **Authoritative Schema**: The single source of truth for all simulation parameters is `src/pasim/config/schema.py`. This module uses `Pydantic` to define a hierarchical `SimulationConfig` model, which ensures that any given configuration is not only structurally correct but also logically valid (e.g., probabilities sum to 1.0, specified regions are valid).

-   **Validated Parameter Access**: The main simulation entry point, `run_genealogy_generator`, validates the user-provided parameters against the `SimulationConfig` model at the very beginning of a run. The rest of the simulation then interacts with the resulting type-safe `config` object, not raw dictionaries, preventing a large class of potential errors.

-   **Historical Drivers**: The configuration schema makes a clear distinction between three types of historical drivers:
    1.  **Historical Shocks** (e.g., `persecutions`): Discrete, instantaneous events that "shock" the system, such as a persecution that destroys a fraction of manuscripts at a specific time.
    2.  **Environmental Regimes** (e.g., `material_transitions`, `script_transitions`): Long-term, evolving environmental conditions that affect the properties of **newly created** entities. This is used to model gradual shifts in technology or culture.
    3.  **Structural Drivers** (e.g., `demand_schedule`): Core inputs that drive the simulation's fundamental mechanics. The `demand_schedule` now specifies *aggregate* demand across all regions for specific ticks. The simulation internally distributes this demand deterministically across regions based on historical allocation rules and a ceiling rounding mechanism. If a tick is not explicitly defined, the last known aggregate demand value is used, ensuring continuity.

## Dependency Management

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging. Poetry helps to manage project dependencies in a more robust and reproducible way compared to traditional `pip` and `requirements.txt`.

### Installation

If you don't have Poetry installed, you can install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

### Installing Project Dependencies

Once Poetry is installed, navigate to the project root directory and run the following command to install all project dependencies:

```bash
poetry install
```

This command will create a virtual environment (if one doesn't already exist) and install all the main and development dependencies specified in `pyproject.toml`.

## Execution

The `pasim` framework provides a robust and flexible execution layer for running both single simulations and complex Monte Carlo experiments.

For detailed instructions on setting up and running experiments, including the experiment directory structure, minimal single-run setup, and the structure of parameter files, please refer to:
-   **`docs/experiments.md`**: Comprehensive guide to experiment configuration and execution.
-   **`experiments/params_template.yaml`**: A canonical template for defining experiment parameters.

### Experiment-Level Execution

The primary entry point for orchestrating full experiments is the `pasim.execution.orchestrator.run_experiment` function. This function takes a single parameter file and manages the entire execution lifecycle, including:

-   Loading and validating experiment parameters (`n_runs`, `max_retries`, `seed`).
-   Launching multiple independent simulation runs in parallel.
-   Implementing robust retry logic for individual run failures.
-   Persisting experiment-level metadata (`experiment_metadata.json`) for overall tracking and reproducibility.
-   Returning a concise summary of the experiment's outcome.

```python
from pasim.execution.orchestrator import run_experiment

# Define your experiment in a YAML file, e.g., experiments/my_experiment/params.yaml
# with fields like:
# n_runs: 100
# max_retries: 3
# seed: 12345

experiment_summary = run_experiment("experiments/my_experiment/params.yaml")
print(experiment_summary)
```

### Parallel Run Orchestration

The `pasim.execution.parallel.run_parallel` function underlies `run_experiment`. It is responsible for:

-   Launching N independent simulation runs concurrently using process-based parallelism.
-   Ensuring each run receives a unique, derived seed for natural RNG variability.
-   Implementing a retry mechanism for individual runs, allowing failures to be recorded and retried without halting the entire batch.
-   Collecting results from all runs (including failure records) and returning a comprehensive summary.

### Single-Run Execution

The lowest-level execution entry point is the `pasim.execution.runner.run_single` function. This function executes a single, in-memory simulation run and automatically persists its results to a unique run directory. It is primarily called by `run_parallel` for each individual simulation.

```python
from pasim.execution.runner import run_single

# Note: For typical experiment execution, you should use `run_experiment`
# with a `params.yaml` that sets `n_runs: 1`. Direct calls to `run_single`
# are usually for debugging or integrating into custom workflows.

result = run_single("experiments/exp001_baseline/params.yaml", seed=42)
```

This function takes two arguments:
-   `params_path` (str): The path to a YAML configuration file for the experiment. This file must be located within the `experiments/` directory.
-   `seed` (int): An optional integer seed for the random number generator to ensure reproducibility.

### Simulation Result

The `run_single` function returns a `SimulationResult` dataclass object, which contains the complete output of the simulation run:

-   `state` (`GenerationState`): The final state of the simulation, including all registries and a `telemetry` list with per-tick data (e.g., number of alive manuscripts).
-   `graph` (`nx.DiGraph`): The complete genealogy graph generated by the simulation.
-   `config` (`SimulationConfig`): The validated Pydantic configuration object used for the run.
-   `seed` (int): The seed used for the run.

## Inspection

After a simulation has been run via `run_single`, the resulting `SimulationResult` object can be passed to a suite of read-only helper functions in the `pasim.analysis.inspection` module. These functions provide convenient, high-level views of the simulation's state for debugging and analysis.

-   **`manuscript_table(state)`**: Returns a list of dictionaries, each representing a manuscript with its complete properties, including its alive/dead status, script, and reputation.
-   **`node_table(state)`**: Returns a list of dictionaries, each representing a node (witness instance) in the genealogy graph, including its ID, associated IDs, and its parent/child relationships.
-   **`genealogy_edges(state)`**: Returns a simple list of `(parent, child)` tuples representing the edges of the genealogy graph.
-   **`lineage_texts(state, leaf_instance_id)`**: Traces the ancestry of a given leaf node back to its root, returning the sequence of texts in chronological order. *(Note: As textual content is not yet stored, this currently returns a list of `None` placeholders).*
-   **`to_networkx_copy(state)`**: Returns a full, deep copy of the `networkx.DiGraph` object, allowing for safe, independent analysis and manipulation.

## Inspection

After a simulation has been run via `run_single`, the resulting `SimulationResult` object can be passed to a suite of read-only helper functions in the `pasim.analysis.inspection` module. These functions provide convenient, high-level views of the simulation's state for debugging and analysis.

-   **`manuscript_table(state)`**: Returns a list of dictionaries, each representing a manuscript with its complete properties, including its alive/dead status, script, and reputation.
-   **`node_table(state)`**: Returns a list of dictionaries, each representing a node (witness instance) in the genealogy graph, including its ID, associated IDs, and its parent/child relationships.
-   **`genealogy_edges(state)`**: Returns a simple list of `(parent, child)` tuples representing the edges of the genealogy graph.
-   **`lineage_texts(state, leaf_instance_id)`**: Traces the ancestry of a given leaf node back to its root, returning the sequence of texts in chronological order. *(Note: As textual content is not yet stored, this currently returns a list of `None` placeholders).*
-   **`to_networkx_copy(state)`**: Returns a full, deep copy of the `networkx.DiGraph` object, allowing for safe, independent analysis and manipulation.

### Test-Time Text Visualization Layer

To enhance debug observability during testing, the simulator includes a test-time text visualization layer. This utility allows developers to visually inspect how textual content (1D NumPy integer arrays) changes between parent and child witness instances during copying events.

-   **Purpose**: Provides inline diff marking in the console for each parent-child copy event, making it easy to see where substitutions occurred and the extent of mutation. This is a debugging aid, not part of the core simulation logic or production output.
-   **Functionality**:
    -   Iterates over every edge in the simulation's `state.graph` (representing a parent → child copy).
    -   Fetches the respective texts via `state.registries.instance_texts`.
    -   Prints the parent and child instance IDs, their full text, and a `Diff` line with `^` markers indicating differing positions.
    -   Automatically truncates long texts (over 120 tokens) to show the head and tail, while still marking differences within the visible sections.
    -   Ensures deterministic output by sorting edges based on the child's birth tick.
-   **Usage**: This feature is integrated directly into the test framework. To enable it, run pytest with the `--show-text-diffs` flag:
    ```bash
    poetry run pytest --show-text-diffs
    ```
    The visualization will automatically appear in the console after relevant tests have executed their simulations.

## Research Data Persistence

To facilitate rigorous academic analysis and ensure full reproducibility, the `pasim` framework includes a comprehensive research-grade persistence layer. Every execution of `run_single()` automatically generates a unique, timestamped run directory (`experiments/<experiment_name>/runs/<run_number>/`) containing a complete snapshot of the simulation's output.

This persistence layer saves all critical simulation artifacts as structured, human-readable plain text files, enabling researchers to:

-   **Reconstruct any experiment**: By saving the exact `config.yaml` used for the run.
-   **Trace the full textual genealogy**: Through `genealogy.json` (graph structure) and `instances.json` (witness instance metadata).
-   **Inspect manuscript lifecycles**: Using `manuscripts.json` (full manuscript registry).
-   **Analyze textual evolution**: With `instance_texts.tsv` providing the complete content of every generated text.
-   **Monitor simulation dynamics**: Via `telemetry.json` (per-tick metrics) and `events.log` (chronological log of key simulation events).

This ensures that all simulation results are fully transparent, auditable, and ready for publication.

## Architecture and Structure

The project `pasim` is designed as a modular framework, providing a structured approach to building and running scientific simulations. The core components are organized within the `src/pasim` directory, with distinct responsibilities:

### Simulation Dynamics: Shocks vs. Environments

The simulation distinguishes between two fundamental types of dynamics, which allows for a nuanced and modular approach to modeling historical change:

-   **Historical Events (Shocks)**: These represent discrete, time-bound, and often global "shocks" or transitions that are imposed on the simulation from the outside. Managed by the **Temporal-Historical Rule Engine** (`historical_events.py`), these events model exogenous phenomena that cause immediate changes to existing entities. For example, a `PersecutionEvent` can be configured to destroy a certain proportion of manuscripts in a specific region and time period. This plug-in architecture allows researchers to easily add, remove, or modify historical scenarios without altering the core simulation loop.

-   **Transition Regimes (Environments)**: These represent persistent, long-term environmental conditions that evolve over time. They are modeled as time-dependent probability distributions that affect **only newly created entities**, not existing ones. This mechanism is ideal for simulating gradual cultural or technological shifts.
    -   The `MaterialTransitionManager` (`material_transition_manager.py`) governs the probability of a new manuscript being made of a certain material (e.g., papyrus, parchment).
    -   The `ScriptTransitionManager` (`script_transition_manager.py`) governs the probability of a new witness being written in a certain script (e.g., uncial, minuscule).

This core distinction—shocks that alter existing state versus environments that shape new state—provides a powerful and flexible framework for constructing complex historical simulations.

-   **Mechanistic Rules**: These are the continuous, local, and emergent processes that drive the simulation forward tick by tick. They include manuscript death, migration, and demand-based spawning (copying). These rules are fundamental to the simulation's engine and are applied consistently. They operate within the context of the environments set by the transition regimes.

-   **`core/`**: This module is intended to house the core, pure, and deterministic simulation logic. This includes the fundamental rules and processes governing the textual transmission model, independent of I/O or execution concerns.
    -   `exemplar_selection.py`: Implements the two-stage exemplar selection logic, which combines geographical proximity with textual authority (reputation) to choose parents for new manuscripts.
    -   `scribal_rules.py`: Implements the composite scribal rule, which is the high-level pipeline for generating a new textual witness. It composes the base transmission, reputation-to-error-intensity, and mutation modules to simulate the full act of copying.
    -   `genealogy.py`: Defines the structural genealogy of witness instances using a `networkx` directed acyclic graph (DAG). It tracks ancestry, timing, and topology (who copied from whom and when).
    -   `genealogy_generator.py`: Orchestrates the deterministic, tick-based generation of the genealogy graph. It uses a state-machine-like process, advancing tick by tick and hosting the injection of scientific rules. It now manages the lifecycle of manuscripts, including their deterministic death, migration, and demand-based spawning. The `GenerationState` dataclass and its `initialise_generation_state` function have been moved to `simulation_state.py` to resolve circular dependencies.
    -   `historical_events.py`: Provides the framework for the **Temporal-Historical Rule Engine**, including the base `HistoricalEvent` abstraction, the `HistoricalEventManager` for dispatching events, and concrete implementations like `PersecutionEvent` (modeling mass destruction of manuscripts). It now directly imports `GenerationState` from `simulation_state.py`.
    -   `simulation_state.py`: Defines the `GenerationState` dataclass, which encapsulates the complete state for a genealogy generation run, along with the `initialise_generation_state` function. This module centralizes the definition of the core simulation state and resolves circular import dependencies with `genealogy_generator.py` and `historical_events.py`.
    -   `tagged_string_constraints.py`: Defines the legal "alphabet" for segment values in a tagged string. It provides a single, authoritative source of truth for the valid state space, ensuring all mutation and factory functions operate within explicitly enforced boundaries.
    -   `transmission.py`: Defines the base rules for how a new tagged string is generated from parent exemplars (e.g., majority voting). This represents the ideal, error-free copying process before any scribal mutations are applied.
    -   `reputation.py`: Defines the policy layer that maps a witness's reputation level to an expected proportion of segments that will mutate during copying. It is also responsible for sampling new reputation scores from a configurable probability distribution. This translates the abstract concept of reputation into a concrete error intensity.
    -   `mutation.py`: Defines the mechanical operator that applies scribal mutations to a tagged string. It takes an expected proportion of segments to change and, under controlled randomness, alters the required number of segments to other legal values.
    -   `rng.py`: Implements the centralized random number generation (RNG) factory. This factory ensures full reproducibility across runs and safe parallel execution by managing `numpy.random.Generator` and `numpy.random.SeedSequence` objects, derived from a single batch-level seed.
    -   `material_transition_manager.py`: Manages the time-dependent probability distribution for new manuscript materials (e.g., papyrus, parchment, paper). This is a persistent rule, not a historical event, affecting only newly spawned manuscripts to reflect historical technological shifts.
    -   `script_transition_manager.py`: Manages the time-dependent probability distribution for new witness scripts (e.g., uncial, minuscule). This rule affects only newly created witnesses to model the gradual evolution of script styles.
    -   `spatial.py`: Provides utilities for handling spatial assignments of manuscripts, ensuring region-specific coordinate generation and maintaining the architectural separation of spatial properties from genealogy nodes.
    -   `state.py`: Defines the core data structures and identity registries for the simulation. This includes `Manuscript` and `Witness` data classes, as well as the `ManuscriptRegistry` and `WitnessRegistry` that manage the unique identities of these entities. A top-level `StateRegistry` class holds these registries, providing a consistent snapshot of what exists in the simulation.
-   **`config/`**: Manages the configuration and parameters for the simulations.
    -   `schema.py`: Defines the complete, hierarchical validation schema for all simulation parameters using `Pydantic`. This is the single source of truth for configuration.
-   **`execution/`**: Handles the orchestration and running of simulations, forming the core of the experiment execution layer.
    -   **`orchestrator.py`**: Implements the high-level `run_experiment` function, serving as the experiment-level entrypoint. It loads experiment parameters, manages experiment metadata (`experiment_metadata.json`), invokes the parallel runner, and returns a comprehensive summary of the experiment's outcome, including retry policies and failure records.
    -   **`parallel.py`**: Provides the `run_parallel` function for orchestrating multiple independent simulation runs concurrently. It ensures process-based parallelism, generates unique seeds for each run, implements robust retry logic with failure handling, and collects run-level results.
    -   **`runner.py`**: Contains the `run_single` function, which is the entry point for executing a single, in-memory simulation run. It is responsible for setting up the RNG, running the core genealogy generation, and persisting the individual run's results.
    -   `batch.py`: (Intended for future batch management, currently unused)
-   **`analysis/`**: Provides tools for processing, analyzing, and visualizing simulation results.
    -   `metrics.py`: For defining and calculating simulation metrics.
    -   `plots.py`: For generating visualizations of results.
    -   `statistics.py`: For statistical analysis of simulation outputs.
-   **`io/`**: Manages input and output operations, including reading initial data and writing simulation results.
    -   `persistence.py`: Provides a research-grade persistence layer for saving both individual simulation run outputs and experiment-level metadata.
        -   For individual runs: Every `run_single()` execution automatically generates a unique, timestamped run directory (`experiments/<experiment_name>/runs/<run_number>/`) containing a complete snapshot of the simulation's output. This process includes a concurrent-safe mechanism for resolving unique run directory names. Artifacts saved per run include:
            -   `config.yaml`: An exact copy of the input configuration file.
            -   `run_metadata.json`: High-level metadata about the run (seed, final tick, total instances/manuscripts, graph nodes/edges).
            -   `genealogy.json`: The complete genealogy graph structure (nodes with instance_id, manuscript_id, birth_tick, reputation; and edges with parent/child relationships).
            -   `instances.json`: Detailed metadata for all witness instances.
            -   `manuscripts.json`: Full details of all manuscripts, including their physical attributes.
            -   `instance_texts.tsv`: A tab-separated file containing the textual content of all witness instances, ordered by birth_tick, ensuring streaming writes for large datasets.
            -   `telemetry.json`: The raw telemetry log from the simulation.
            -   `events.log`: A chronological plain-text log of key simulation events (manuscript creation/death, instance creation with parent info).
        -   For experiments: The `experiment_metadata.json` file is created at the experiment root (`experiments/<experiment_name>/`) to provide a durable record describing the experiment as a whole. This includes its parameters, execution status, overall run counts (successful, failed, retried), and timestamps.    -   `formats.py`: For handling different data formats.
    -   `output.py`: For writing simulation outputs.
    -   `aggregation.py`: For aggregating results from multiple runs.
-   **`utils/`**: Contains common utility functions used across the project.
    -   `logging.py`: For logging mechanisms.
    -   `timing.py`: For performance measurement.
-   **`cli.py`**: The command-line interface entry point for interacting with the simulation framework.

## General Functioning (Planned)

The `pasim` framework will facilitate the following general workflow for researchers:

1.  **Configuration**: Users will define simulation parameters (e.g., number of scribes, error rates, manuscript branching, **migration probabilities**, simulation duration, **regional demand for new manuscripts**, historical events like `persecutions`, and material transition schedules like `material_transitions`) using configuration files, likely validated by the `config/schema.py`. Manuscript lifespans are now probabilistically determined at creation based on material and region, rather than being supplied as a pre-generated list.
2.  **Initialization**: The simulation will initialize its state based on an initial text (the Pericope Adulterae) and the specified configuration.
3.  **Execution**: The `execution/` module will manage the running of Monte Carlo simulations. This involves iteratively applying mechanistic rules (like death and migration), processing exogenous historical events, and dynamically spawning new manuscripts based on regional demand. For each new witness instance, its textual content is generated either as an initial autograph text (using `text_initialisation.py`) or by copying and mutating its exemplar(s) through the `scribal_rules` pipeline. All stochasticity is controlled via `core/rng.py` to ensure reproducibility.
4.  **Data Collection**: Throughout the simulation, relevant data points (e.g., changes in text, manuscript lineages, manuscript population dynamics, **and the textual content of each witness instance via `state.registries.instance_texts`**) will be collected and stored via the `io/` module.
5.  **Analysis**: After simulation completion, the `analysis/` module will be used to process the collected data, calculate metrics, generate plots, and perform statistical analyses to derive insights into textual transmission.
6.  **Reporting**: Results and analyses will be outputted in various formats for researchers to study.

This setup enables researchers to explore various hypotheses about the textual history of the Pericope Adulterae by adjusting simulation parameters and observing the emergent textual traditions.

## Testing

### `tests/test_historical_dynamics.py`

This test module ensures that the historical and temporal systems of the simulation behave correctly and reproducibly. It contains a suite of tests that validate the determinism, correctness, and isolation of the various dynamic components of the simulation engine.

These tests guarantee that historical dynamics are deterministic, parameter-driven, and isolated from core mechanics. They verify that scheduled events (like persecutions) and environmental transitions (like material or script usage) behave correctly and reproducibly.

-   **Persecution Correctness Test**: Verifies that a persecution event correctly removes a specified proportion of manuscripts from the alive set within a targeted region, without affecting manuscripts in other regions or the integrity of the genealogy graph.
-   **Persecution Determinism Test**: Ensures that running a simulation with a persecution event multiple times with the same seed produces the exact same set of destroyed manuscripts, and that different seeds lead to different outcomes.
-   **Material Transition Test**: Validates that newly spawned manuscripts are assigned materials according to the time-dependent probability distribution defined in the `MaterialTransitionManager`. It asserts that manuscripts created before and after a transition point have the expected materials and that existing manuscripts' materials are not altered.
-   **Script Transition Test**: Similar to the material transition test, this verifies that newly created witness instances are assigned scripts (e.g., uncial, minuscule) based on the active schedule in the `ScriptTransitionManager`.
-   **Demand Schedule Test**: Confirms that the simulation spawns new manuscripts to meet the minimum numbers specified in the regional demand schedule and that the logic for retrieving demand correctly falls back to the last known value for ticks not explicitly defined.
-   **Migration Determinism Test**: Guarantees that the stochastic process of manuscript migration (both between and within regions) is fully deterministic for a given seed, producing identical migration histories across identical runs.
-   **Event Ordering Stability Test**: Checks that the `HistoricalEventManager` correctly processes events based on their `start_tick`, regardless of their order in the configuration file, ensuring stable and predictable outcomes.
-   **No Side-Effect Tests**: Verifies that manager components (`HistoricalEventManager`, `MaterialTransitionManager`, `ScriptTransitionManager`) do not cause unintended side effects, such as modifying the simulation's tick or state, when they are called.

These tests are crucial for ensuring the scientific validity and reproducibility of the simulation, providing confidence that the results are a direct consequence of the configured parameters and not artifacts of implementation errors.

### `tests/test_persistence.py`

This module contains a comprehensive test suite for the research data persistence layer, ensuring its correctness, determinism, and integrity. These tests validate that:

-   Run directories are created correctly, following a numbered sequence (e.g., `1`, `2`, `3`).
-   All required output files (`config.yaml`, `run_metadata.json`, `genealogy.json`, `instances.json`, `manuscripts.json`, `instance_texts.tsv`, `telemetry.json`, `events.log`) are present after a simulation run.
-   The contents of `run_metadata.json` are consistent with the simulation's `SimulationResult` object (seed, graph nodes/edges, total instances/manuscripts, final tick).
-   The `genealogy.json` file accurately reflects the in-memory genealogy graph in terms of node and edge counts, and references valid node IDs.
-   `instance_texts.tsv` correctly stores textual data, with an appropriate header, the correct number of rows, integer tokens, and content matching in-memory NumPy arrays for selected instances.
-   `telemetry.json` precisely matches the in-memory telemetry data from the simulation.
-   `events.log` contains comprehensive coverage of key simulation events, including instance births, manuscript births, and manuscript deaths.
-   The persistence process itself does not mutate the in-memory simulation state, ensuring data integrity before and after saving.


## Detailed Module and File Summaries

### `src/pasim/__init__.py`

This file is an empty `__init__.py` file, serving primarily to mark the `pasim` directory as a Python package. It does not contain any functional code or exposed modules.

### `src/pasim/analysis/__init__.py`

This file is an empty `__init__.py` file, serving primarily to mark the `analysis` directory as a Python subpackage within `pasim`. It does not contain any functional code or exposed modules.

### `src/pasim/analysis/inspection.py`

This module provides a suite of read-only helper functions for inspecting the state of a completed simulation run. It includes utilities for generating tabular views of manuscripts and nodes, extracting the genealogy graph structure, and tracing textual lineages. These functions are pure and do not modify the input state.

### `src/pasim/analysis/metrics.py`

This file is currently empty. It is intended to house functions and classes related to defining and calculating simulation metrics, as outlined in the `project_overview.md`.

### `src/pasim/analysis/plots.py`

This file is currently empty. It is intended to house functions for generating visualizations of simulation results, as outlined in the `project_overview.md`.

### `src/pasim/analysis/statistics.py`

This file is currently empty. It is intended to house functions for performing statistical analysis of simulation outputs, as outlined in the `project_overview.md`.

### `src/pasim/cli.py`

This module provides the comprehensive command-line interface (CLI) for the `pasim` project, built using `argparse`. It serves as the primary entry point for users to interact with the simulation framework without direct Python scripting.

**Key Features and Commands:**

*   **`pasim run <params_path>`**: Executes a simulation experiment using a specified parameters YAML file. It leverages `pasim.execution.orchestrator.run_experiment` for robust execution, parallel processing, and result persistence. Supports a `--verbose` flag for detailed logging.
*   **`pasim reset [--force]`**: Cleans up experiment-generated data by removing `runs/` directories and `experiment_metadata.json` files from experiment folders. It prompts for user confirmation unless the `--force` flag is used.
*   **`pasim list`**: Scans the `experiments/` directory and provides a summary of all defined experiments, including their names, requested runs, completed runs, and current status based on metadata.
*   **`pasim tests`**: Programmatically invokes `pytest` to run the project's test suite. It can forward a `--verbose` flag to `pytest`.
*   **`pasim help` / `pasim --help`**: Displays detailed usage instructions for the CLI, including available commands, global flags, and information about the experiment directory structure.

The CLI is designed with clean error handling, providing informative messages and exiting with non-zero status codes on failure. It integrates with the project's logging system for configurable output verbosity.

### `src/pasim/config/__init__.py`

This file is an empty `__init__.py` file, serving primarily to mark the `config` directory as a Python subpackage within `pasim`. It does not contain any functional code or exposed modules.

### `src/pasim/config/schema.py`

This module defines the comprehensive, hierarchical configuration schema for `pasim` simulations using Pydantic. It acts as the single source of truth for validating simulation parameters, ensuring structural correctness and logical validity (e.g., probabilities summing to 1.0, sequential `start_tick` values).

**Key Pydantic Models:**

*   **`PersecutionEventConfig`:** Configures a single historical "shock" event.
    *   `start_tick`, `end_tick`: Define the temporal scope of the event.
    *   `regions`: A list of region names where the persecution applies, validated against `pasim.core.state.Region` enum.
    *   `kill_proportion`: The proportion of manuscripts to destroy (0.0 to 1.0).
    *   Includes validators to ensure valid regions and correct tick ordering.
*   **`MaterialTransitionConfig`:** Configures a point in time where the probability distribution for new manuscript materials changes.
    *   `start_tick`: The tick at which this distribution becomes active.
    *   `distribution`: A dictionary mapping `Material` names (e.g., 'papyrus', 'parchment') to their probabilities, validated to sum to 1.0 and use valid `Material` enum values.
*   **`ScriptTransitionConfig`:** Similar to `MaterialTransitionConfig`, but for `Script` types (e.g., 'uncial', 'minuscule').
    *   `start_tick`: The tick at which this distribution becomes active.
    *   `distribution`: A dictionary mapping `Script` names to their probabilities, validated to sum to 1.0 and use valid `Script` enum values.
*   **`DemandScheduleConfig`:** Defines the demand for new manuscripts across different regions over time.
    *   `__root__`: A dictionary where keys are ticks (integers) and values are dictionaries mapping `Region` enum objects to the integer demand count for that tick. The schema validator automatically converts string region names from the config file into `Region` enums.
    *   Includes validators for tick values, region names, and demand counts.
*   **`SimulationConfig`:** The root model encompassing all simulation parameters.
    *   `total_ticks`: The total duration of the simulation.
    *   `text_length`: The length of the tagged string used for textual content.
    *   `p_region_migration`: Probability of a manuscript migrating to a different region.
    *   `p_internal_relocation`: Probability of a manuscript relocating within its current region.
    *   `reputation_distribution`: A dictionary mapping reputation scores (1-5) to their probabilities.
    *   Includes lists of `PersecutionEventConfig`, `MaterialTransitionConfig`, and `ScriptTransitionConfig` objects.
    *   `demand_schedule`: An instance of `DemandScheduleConfig`.
    *   Includes validators to ensure `start_tick` values in transition schedules are strictly increasing.

**Utility Functions:**

*   **`get_persecution_events(params: dict)`:** Extracts and validates persecution event configurations from a raw parameter dictionary.
*   **`get_material_schedule(params: dict)`:** Extracts and validates material transition schedules.
*   **`get_script_schedule(params: dict)`:** Extracts and validates script transition schedules.
*   **`get_demand_for_tick(params: dict, tick: int)`:** Returns the demand for a specific tick, implementing logic to use the last known demand value if the current tick is not explicitly defined in the schedule.

This module is crucial for ensuring that all simulations run with well-formed and logically consistent parameters, preventing a large class of potential errors.

### `src/pasim/core/__init__.py`

This file is an empty `__init__.py` file, serving primarily to mark the `core` directory as a Python subpackage within `pasim`. It does not contain any functional code or exposed modules.

### `src/pasim/core/exemplar_selection.py`

This module, `exemplar_selection.py`, provides the `select_exemplars` function, which implements the crucial logic for choosing parent exemplars (source manuscripts) for a newly spawned manuscript. The selection process is designed as a two-stage filter, prioritizing both geographical proximity and textual authority (reputation), reflecting a nuanced model of manuscript transmission.

**Key Functionality and Process (`select_exemplars`):**

1.  **Geographical Filtering**:
    *   Takes `new_manuscript` (the manuscript needing parents) and a list of `alive_manuscripts_in_region` (all active manuscripts in the same geographical area).
    *   Calculates the Euclidean distance between the `new_manuscript`'s location and each `alive_manuscript_in_region`'s location.
    *   Selects the up to 10 closest manuscripts based on this distance, ensuring that local transmission is prioritized.
2.  **Mapping to Witness Instances**:
    *   Converts the selected `closest_manuscripts` (physical `Manuscript` objects) into their corresponding `WitnessInstanceID`s, using the `manuscript_to_instance_map`. These witness instances, which are nodes in the genealogy graph, are the actual entities that will serve as exemplars.
3.  **Reputation-based Ranking**:
    *   Retrieves the `reputation` attribute from each candidate `WitnessInstance` node in the `graph`.
    *   Sorts the `candidate_instances` in descending order based on their reputation, giving preference to more authoritative texts.
4.  **Final Selection**:
    *   Randomly determines the number of exemplars (`n`) to select from a predefined distribution (currently 80% chance for 1 exemplar, 10% for 2, 10% for 3).
    *   Selects the top `n` `WitnessInstanceID`s from the reputation-sorted list as the final exemplars.

**Helper Function:**

*   **`_euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float`:** A private utility function to calculate the standard Euclidean distance between two 2D points.

**Separation of Concerns:**

This module distinctly applies geographical constraints at the manuscript level before considering textual authority at the witness instance level, adhering to the project's architectural principle of separating physical manuscript properties from abstract genealogical structure.

### `src/pasim/core/genealogy.py`

This module provides the fundamental building blocks for constructing and managing the `pasim` simulation's genealogy graph, which models the structural and temporal relationships between witness instances. It uses the `networkx` library to represent this genealogy as a Directed Acyclic Graph (DAG), where nodes are witness instances and edges represent copying events.

**Core Principles:**

*   **Topology-focused**: This module is strictly concerned with the graph's topology (identities, ancestry, timing of creation) and explicitly *excludes* any knowledge of the textual content ("tagged strings") of the witnesses. This separation allows the genealogy to serve as a fixed scaffold for various textual evolution models.
*   **Determinism**: Functions are designed to be deterministic, ensuring reproducible graph construction given the same inputs.

**Key Components and Functions:**

*   **`GenealogyGraph` (Type Alias):** Simply an alias for `networkx.DiGraph`.
*   **`GenealogyValidationError`:** A custom exception raised when any genealogy invariant is violated.
*   **`create_empty_genealogy() -> GenealogyGraph`:** Returns an empty `networkx.DiGraph` instance, ready for population.
*   **`add_root_node(...)`:** Adds a root witness instance to the graph (a node with no parents). This function strictly enforces that a root node can only be added to an empty graph, ensuring **exactly one autograph** for the entire genealogy. It calls `validate_genealogy` after addition.
    *   Takes `graph`, `node_id` (unique identifier), `witness_id` (foreign key to `Witness` registry), `manuscript_id` (foreign key to `Manuscript` registry), `birth_tick`, `reputation`, and an optional `death_tick` as attributes for the node.
    *   Raises `ValueError` if `node_id` already exists.
    *   Raises `GenealogyValidationError` if a root node is attempted on a non-empty graph.
*   **`add_child_node(...)`:** Adds a new witness instance node as a child of one or more existing parent nodes, representing a copying event. This function now explicitly checks for and prevents the creation of cycles or other invariant violations by calling `validate_genealogy` after the node and edges are added, reverting the changes if validation fails.
    *   Takes similar parameters to `add_root_node`, plus `parent_node_ids` (a list of node IDs for the parent(s)).
    *   Adds edges from each parent to the new child.
    *   Raises `ValueError` if a node with `node_id` already exists, if any parent ID does not exist, or if no parents are provided.
    *   Raises `GenealogyValidationError` if adding the node violates any genealogy invariants (e.g., introduces a cycle, or implicitly creates a second root by attempting to add a node without parents when the graph is not empty).
*   **`validate_genealogy(graph: GenealogyGraph) -> None`:** This is the **single authoritative validator** for genealogy correctness. It performs structural sanity checks and enforces critical invariants:
    *   The graph must be a `networkx.DiGraph` and a **Directed Acyclic Graph (DAG)**.
    *   There must be **at most one root (autograph)**. This implicitly ensures **no orphan instances** (every non-autograph node has at least one parent) by disallowing multiple disconnected components, each with its own root.
    *   All nodes must contain essential attributes (`witness_id`, `manuscript_id`, `birth_tick`, `reputation`).
    *   Raises `TypeError` if the graph is not a `networkx.DiGraph`.
    *   Raises `GenealogyValidationError` if any invariant is violated.
*   **Query Functions:**
    *   **`get_parents(graph, node_id) -> List[NodeID]`:** Returns a list of parent node IDs for a given node.
    *   **`get_children(graph, node_id) -> List[NodeID]`:** Returns a list of child node IDs for a given node.
    *   **`get_roots(graph) -> List[NodeID]`:** Returns all nodes in the graph that have no incoming edges.
    *   **`get_leaves(graph) -> List[NodeID]`:** Returns all nodes in the graph that have no outgoing edges.

This module is fundamental for representing the lineage and historical development of textual witnesses, forming the backbone upon which textual variations and other simulation aspects are built.

### `src/pasim/core/genealogy_generator.py`

This module, `genealogy_generator.py`, is the central orchestrator for the simulation's core dynamic: generating the manuscript genealogy graph. It implements a tick-based simulation loop, advancing the simulation clock and managing a series of discrete stages within each tick to ensure determinism and logical consistency.

**Key Components and Functions:**

*   **`GenerationState` dataclass:** Encapsulates the complete state of the genealogy generation, including the current `tick`, the `networkx` graph (`graph`), `StateRegistry` (for manuscripts and witnesses), `alive_manuscripts`, and a mapping from manuscript IDs to witness instance IDs (`manuscript_to_instance_map`). It also holds counters for generating unique IDs for manuscripts, witnesses, and witness instances.
*   **`initialise_generation_state()`:** Creates and returns an empty `GenerationState` object, setting the initial tick to zero and preparing empty data structures.
*   **`handle_deaths(state: GenerationState)`:** Identifies and removes manuscripts whose `death_tick` has arrived from the `alive_manuscripts` set. This affects their availability for future copying but does not alter the historical record in the graph or registry.
*   **`handle_migration(state: GenerationState, rng: RNG, p_region_migration: float, p_internal_relocation: Optional[float])`:** Manages the movement of manuscripts. It applies probabilities for manuscripts to migrate between different regions or relocate within their current region, updating their `region` and `location` attributes.
*   **`_spawn_new_manuscripts_from_demand(...)`:** This private function is responsible for creating new manuscripts to meet regional demand. If the number of alive manuscripts in a region is below the `demanded_count`, new manuscripts are spawned. For each new manuscript, an associated `Witness` and `WitnessInstance` (graph node) are created. It handles the assignment of a `material` (via `MaterialTransitionManager`), `reputation`, and `script` (via `ScriptTransitionManager`). The `death_tick` is now probabilistically determined at this stage, based on the manuscript's material and region, using the `lifespan` module. Critically, if `select_exemplars` returns no suitable parents (e.g., no alive manuscripts in the region), and the graph is not empty (i.e., not the very first autograph), it will **randomly select parents from all currently alive instances** to ensure the new manuscript is always a child node, thereby maintaining the **single autograph invariant**.
*   **`advance_tick(...)`:** This is the core per-tick orchestrator. It increments the simulation clock and calls the other stage-specific functions in a strict order: `handle_deaths`, `event_manager.apply_events_for_tick` (historical events), `handle_migration`, and `_spawn_new_manuscripts_from_demand`.
*   **`run_genealogy_generator(parameters: Dict[str, Any], rng: RNG)`:** The high-level entry point for the genealogy generation. It validates input parameters against `SimulationConfig`, initializes the simulation state and managers (including `HistoricalEventManager`, `MaterialTransitionManager`, and `ScriptTransitionManager`), and then runs the main simulation loop by iteratively calling `advance_tick` for the specified `total_ticks`. It ensures reproducibility by using a seeded NumPy random number generator.

**Separation of Concerns:**

The module emphasizes a clear separation between the abstract `Witness Instance` (nodes in the `networkx` graph representing lineage) and the concrete `Manuscript` (physical objects with rich metadata like region, material, birth/death ticks).

**Exclusions:**

This module explicitly excludes exemplar selection policies (handled by `exemplar_selection.py`), contamination/scribal error models (handled by `scribal_rules.py`, `transmission.py`, `mutation.py`), and batch execution/file I/O, delegating these responsibilities to other specialized modules.

### `src/pasim/core/historical_events.py`

This module establishes the framework for integrating exogenous historical events, often referred to as "shocks," into the simulation. It distinguishes these events from continuous mechanistic rules by providing a plug-in architecture for discrete, time-bound, and potentially global or regional interventions that modify the simulation state.

**Key Components and Functionality:**

*   **`HistoricalEvent` (Abstract Base Class):**
    *   Defines the common interface for all historical events.
    *   Attributes:
        *   `start_tick` (int): The simulation tick when the event begins.
        *   `end_tick` (Optional[int]): The tick when the event ends. If `None`, the event is instantaneous at `start_tick`.
        *   `regions` (Optional[Set[str]]): A set of region identifiers; if `None`, the event applies globally.
    *   Abstract method `apply(self, state: GenerationState, rng: np.random.Generator)`: Must be implemented by concrete event classes to define the event's effect on the simulation state.
    *   `is_active(self, tick: int)`: Determines if the event is active at a given tick, considering `start_tick` and `end_tick`.
*   **`PersecutionEvent` (Concrete Historical Event):**
    *   A subclass of `HistoricalEvent` that models a "persecution" shock, leading to the destruction of a fraction of manuscripts.
    *   Attribute `kill_proportion` (float): The percentage of eligible manuscripts to destroy (between 0.0 and 1.0).
    *   `__post_init__()`: Validates `kill_proportion` to be within the [0, 1] range.
    *   `apply(...)`:
        *   Identifies eligible manuscripts based on `state.alive_manuscripts` and the event's `regions`.
        *   Calculates the number of manuscripts to destroy based on `kill_proportion`.
        *   Randomly selects victims from the eligible pool using the provided `rng`.
        *   Removes the selected `victims` from `state.alive_manuscripts`, making them unavailable for future copying.
*   **`create_event_from_config(config: Dict[str, Any]) -> HistoricalEvent`:**
    *   A factory function that dynamically creates `HistoricalEvent` objects (e.g., `PersecutionEvent`) from a configuration dictionary, based on an `event_type` field.
*   **`HistoricalEventManager` class:**
    *   Manages a collection of `HistoricalEvent` objects.
    *   `__init__(self, event_configs: Optional[List[Dict[str, Any]]] = None)`:
        *   Initializes the manager by creating event objects from a list of configuration dictionaries using `create_event_from_config`.
        *   Sorts the events deterministically by `start_tick` and event class name.
    *   `apply_events_for_tick(self, state: GenerationState, rng: np.random.Generator)`:
        *   Iterates through all registered events.
        *   Identifies events that are `is_active()` for the current `state.tick`.
        *   Applies each active event to the `state` using its `apply()` method.

This module provides a flexible and extensible mechanism for researchers to define and inject various historical scenarios into the simulation, complementing the continuous mechanistic rules and enabling the study of their impact on textual transmission.

### `src/pasim/core/material_transition_manager.py`

This module implements the `MaterialTransitionManager` class, which is responsible for modeling the historical evolution of writing materials used for manuscripts. It manages a time-dependent schedule that dictates the probability distribution of different materials (e.g., papyrus, parchment, paper) for *newly created* manuscripts as the simulation progresses.

**Key Aspects and Functionality:**

*   **Purpose:** To introduce historical realism by reflecting technological and cultural shifts in manuscript production. It is explicitly *not* a `HistoricalEvent` (which represents discrete "shocks"), but rather a persistent environmental rule that influences the properties of entities at their point of creation. Existing manuscripts are not affected; only new ones spawned by demand are assigned materials based on the current probability distribution.
*   **`MaterialTransitionManager` class:**
    *   **`__init__(self, schedule_configs: List[Dict[str, Any]])`:**
        *   Initializes the manager with a sorted list of `schedule_configs`. Each configuration contains a `start_tick` and a `distribution` (a dictionary mapping material names to probabilities).
        *   Performs comprehensive validation of the input schedule:
            *   Checks for an empty schedule (requires at least one entry, e.g., for `start_tick` 0).
            *   Ensures all material names within the distributions correspond to valid `Material` enum members (from `pasim.core.state`).
            *   Verifies that all probabilities are non-negative.
            *   Confirms that the probabilities for each distribution sum to 1.0 (with a numerical tolerance).
        *   Stores a `_processed_schedule` internally, which contains validated `Material` enum members and their probabilities for efficient runtime use.
    *   **`get_material_for_tick(self, tick: int, rng: np.random.Generator) -> Material`:**
        *   Given a `tick` and a NumPy random number generator (`rng`), it identifies the currently active material probability distribution. This is done by finding the configuration entry with the largest `start_tick` that is less than or equal to the current `tick`.
        *   Uses `rng.choice` to deterministically sample a `Material` enum value from the active distribution.
        *   A `RuntimeError` is raised if no active distribution can be found for a given tick, although proper configuration should prevent this by having an entry for `start_tick` 0.

This manager is essential for dynamically assigning material properties to new manuscripts in a historically plausible manner, contributing to the complexity and realism of the simulation.

### `src/pasim/core/mutation.py`

This module, `mutation.py`, provides the core mechanistic operator for introducing scribal mutations (errors) into a "tagged string," which represents the textual content of a witness. It defines how a textual witness changes during the copying process based on a specified intensity of mutation.

**Key Functionality (`mutate_tagged_string`):**

*   **Purpose:** To simulate the discrete changes that occur when a scribe copies a text, leading to textual variations. It operates on a NumPy array (`NDArray[np.int16]`) representing the tagged string.
*   **Inputs:**
    *   `tagged_string` (NDArray[np.int16]): The input text to be mutated. It is assumed to be valid.
    *   `rng` (`np.random.Generator`): A seeded NumPy random number generator to ensure all stochastic operations are deterministic and reproducible.
    *   `expected_proportion` (float): A value between 0.0 and 1.0 representing the desired proportion of segments (elements) in the tagged string that should be mutated. This effectively controls the "error rate" for a given copying event.
*   **Process:**
    1.  **Validation:** Checks if `expected_proportion` is within the valid range [0.0, 1.0] and if the `tagged_string` has the correct `TAGGED_STRING_LENGTH` (imported from `tagged_string_constraints`).
    2.  **No Mutation Case:** If `expected_proportion` is 0.0 or if `n_mutations` rounds to 0, a copy of the original string is returned without changes.
    3.  **Calculate Number of Mutations:** Determines `n_mutations` by rounding `expected_proportion * TAGGED_STRING_LENGTH`.
    4.  **Select Indices:** Uses `rng.choice` to randomly select `n_mutations` *unique* indices (positions) within the tagged string where mutations will occur.
    5.  **Apply Mutations:** For each selected index:
        *   The current value at that index is retrieved.
        *   `sample_alternative_value` (from `tagged_string_constraints`) is used to generate a *different* legal value for that segment. This ensures a meaningful change and adherence to defined constraints for segment values.
        *   The new value replaces the old one in a `copy()` of the original `tagged_string`. The original input string is not modified.
    6.  **Safety Check:** A final call to `validate_tagged_string` ensures the output string remains valid.
*   **Determinism:** The function guarantees deterministic output for identical inputs and `rng` states.

**Dependencies:**

*   Relies on `pasim.core.tagged_string_constraints` for `TAGGED_STRING_LENGTH`, `sample_alternative_value`, and `validate_tagged_string`, which define the valid state space for textual segments and how alternative values are sampled.

This module is a fundamental part of the scribal rules pipeline, directly responsible for generating textual variations in the simulated transmission process.

### `src/pasim/core/reputation.py`

This module defines the policy layer for translating an abstract "reputation" score of a witness instance into a concrete "expected proportion of segments" that are likely to be mutated during a copying event. It also provides functionality to sample new reputation scores. This module effectively bridges the high-level concept of textual authority with the low-level mechanics of scribal error introduction.

**Key Components and Functions:**

*   **`DEFAULT_REPUTATION_MAPPING` (Dictionary):** A module-level constant defining a default mapping from integer reputation levels (1-5) to an `expected_proportion` of segments that will mutate. This serves as a baseline, where higher reputation might (or might not, depending on the configured mapping) imply a lower mutation rate.
*   **`validate_reputation(reputation: int) -> None`:**
    *   A utility function that validates if a given `reputation` integer is within the legal range of 1 to 5, raising a `ValueError` if it's not.
*   **`expected_mutation_proportion(reputation: int, mapping: Optional[Dict[int, float]] = None) -> float`:**
    *   This is the core policy lookup function. It takes a `reputation` level and an optional `mapping` (which defaults to `DEFAULT_REPUTATION_MAPPING`).
    *   It retrieves the corresponding `expected_proportion` of segments that should mutate for that reputation level.
    *   Performs validation to ensure the input `reputation` is valid, exists in the mapping, and the retrieved proportion is between 0.0 and 1.0.
    *   The function is fully deterministic.
*   **`sample_reputation(rng: np.random.Generator, reputation_distribution: Dict[int, float]) -> int`:**
    *   Allows for probabilistic assignment of reputation scores to new witness instances.
    *   Takes a `rng` (NumPy random number generator) for deterministic sampling and a `reputation_distribution` dictionary, which is expected to be pre-validated by the `SimulationConfig`.
    *   Uses `rng.choice` to sample an integer reputation score (1-5) based on the specified probabilities.

This module plays a critical role in modeling the impact of textual authority on copying fidelity, allowing researchers to explore how perceived quality influences the introduction of textual variations.

### `src/pasim/core/rng.py`

This module provides a centralized and reproducible random number generation (RNG) factory through the `RNGContext` class. Its primary purpose is to ensure determinism and independence of random number streams across multiple simulation runs, particularly in parallel execution environments, all derived from a single master seed.

**Key Concepts and Mechanism:**

*   **Reproducibility:** The entire batch of simulations can be reproduced from a single user-defined integer seed.
*   **Independence:** Each individual simulation run receives its own independent `numpy.random.Generator`, preventing state contention and ensuring results are identical regardless of execution order or degree of parallelism.
*   **Modern NumPy API:** Utilizes `numpy.random.SeedSequence` and `numpy.random.Generator` for robust and efficient RNG management.

**`RNGContext` Class:**

*   **`__init__(self, seed: Optional[int] = None)`:**
    *   Initializes the `RNGContext` with a master `seed`.
    *   If `seed` is `None`, a default fixed seed (`20240105`) is used to ensure reproducibility even without explicit user input.
    *   Creates a root `numpy.random.SeedSequence` from this effective seed. This root sequence is the source for all subsequent child sequences.
*   **`spawn(self, n: int) -> List[np.random.Generator]`:**
    *   This method is the core factory. It takes an integer `n` and returns a list of `n` independent `numpy.random.Generator` instances.
    *   It achieves independence and determinism by using the root `SeedSequence` to `spawn()` `n` child `SeedSequence` objects. Each of these child sequences is then used to initialize a new `numpy.random.Generator`.
    *   Calling `spawn()` multiple times with the same `n` on the same `RNGContext` instance will always produce the identical list of generators.

**Usage Guideline:**

*   Users (other modules in the simulation) should *not* directly use `numpy.random` functions. Instead, they should obtain a `np.random.Generator` instance from `RNGContext` and pass it to functions that require randomness.

This module is critical for maintaining scientific rigor and enabling robust testing and analysis of the simulation's stochastic elements, guaranteeing that experiments can be precisely replicated.

### `src/pasim/core/scribal_rules.py`

This module, `scribal_rules.py`, implements the overarching "composite scribal rule" by orchestrating several lower-level primitives into a coherent pipeline. It simulates the complete process a scribe undergoes when copying a text, from using parent exemplars to introducing mutations, thereby forming the core of the textual evolution model.

**Key Functionality (`apply_scribal_rule`):**

*   **Purpose:** To produce a new, potentially mutated, tagged string for a new witness instance based on its parent exemplars and the scribe's (or witness's) reputation.
*   **Inputs:**
    *   `exemplar_texts` (`List[TaggedString]`): A list of one or more `TaggedString` (NumPy arrays) representing the texts of the parent exemplars.
    *   `rng` (`np.random.Generator`): A seeded NumPy random number generator for all stochastic operations (e.g., tie-breaking in majority voting, mutation selection).
    *   `reputation` (int): The integer reputation level (1-5) of the new witness, which directly influences the expected error intensity.
    *   `mutation_mapping` (Optional[Dict[int, float]]): An optional dictionary to override the default mapping from reputation to mutation proportion.
*   **Scribal Pipeline Stages:**
    1.  **Base Transmission:**
        *   If there's only one exemplar, the `copy_from_single_exemplar` function (from `transmission.py`) is used to create a base text directly.
        *   If there are multiple exemplars, `majority_from_exemplars` (from `transmission.py`) is used. This function combines the texts by applying segment-wise majority voting, with `rng` used for tie-breaking, to produce a single resolved base text.
    2.  **Error Intensity Determination:**
        *   The `expected_mutation_proportion` function (from `reputation.py`) is called with the `reputation` of the new witness and the optional `mutation_mapping`. This translates the abstract reputation score into a concrete `expected_proportion` of segments to be mutated.
    3.  **Mutation:**
        *   Finally, the `mutate_tagged_string` function (from `mutation.py`) is applied to the `base_text`. It uses the `rng` and the `expected_proportion` to introduce the specified number of random changes to the text.
*   **Output:** A `TaggedString` (NumPy array) representing the text of the newly created witness instance after the copying and mutation process.
*   **Determinism:** The function ensures full determinism given consistent inputs and `rng` state.

**Dependencies:**

*   `pasim.core.transmission`: For base text generation from single or multiple exemplars.
*   `pasim.core.reputation`: For determining the expected mutation proportion based on reputation.
*   `pasim.core.mutation`: For applying the actual segment-level mutations.

This module is central to simulating the emergent textual variations in the simulation, integrating policy (reputation) with mechanics (mutation) to model scribal behavior realistically.

### `src/pasim/core/script_transition_manager.py`

This module implements the `ScriptTransitionManager` class, which is responsible for managing the time-dependent probability distribution of script styles for newly created witnesses throughout the simulation. It models gradual historical shifts in script usage (e.g., from uncial to minuscule).

**Key Aspects and Functionality:**

*   **Purpose:** To define environmental regimes that affect the properties of *newly created* entities, specifically the `script` attribute of `Witness` objects. It is distinct from `HistoricalEvent` as it does not represent an instantaneous "shock" but rather a persistent rule influencing continuous processes.
*   **`ScriptTransitionManager` class:**
    *   **`__init__(self, schedule_configs: List[Dict[str, Any]])`:**
        *   Initializes the manager with a sorted list of `schedule_configs`. Each configuration specifies a `start_tick` and a `distribution` (a dictionary mapping script names to probabilities).
        *   Performs rigorous validation during initialization:
            *   Ensures the schedule is not empty.
            *   Validates that all script names in the distributions correspond to valid `Script` enum members (defined in `pasim.core.state`).
            *   Checks that all probabilities are non-negative.
            *   Verifies that the probabilities for each `start_tick` distribution sum to 1.0 (with a small floating-point tolerance).
        *   Stores a `_processed_schedule` containing validated script enum values and their probabilities for efficient lookup.
    *   **`get_script_for_tick(self, tick: int, rng: np.random.Generator) -> Script`:**
        *   Determines the active script probability distribution based on the given `tick`. It finds the latest `start_tick` in the schedule that is less than or equal to the current `tick`.
        *   Uses the provided NumPy random number generator (`rng`) to sample a `Script` enum value from the active distribution.
        *   Ensures deterministic sampling if the `rng` is seeded.

This manager is a crucial component for introducing historical realism into the simulation by modeling evolving cultural or technological influences on manuscript production without altering already existing manuscripts.

### `src/pasim/core/spatial.py`

This module, `spatial.py`, provides utility functions for managing the spatial assignments of manuscripts within the simulation. It reinforces the architectural decision that geographical properties are associated with the physical `Manuscript` objects rather than the abstract genealogy nodes.

**Key Components and Functions:**

*   **`REGION_BOUNDS` (Dictionary Constant):**
    *   Defines the bounding box for each geographical `Region` (from `pasim.core.state.Region`).
    *   Each region is assigned its own independent planar (x, y) coordinate space. Coordinates are *not* comparable across regions (e.g., `(50, 50)` in "Asia Minor" is not geographically related to `(50, 50)` in "Egypt").
    *   The format is `Region: ((xmin, xmax), (ymin, ymax))`.
*   **`generate_random_coordinates(region: Region, rng: RNG) -> Tuple[float, float]`:**
    *   **Purpose:** To deterministically generate random (x, y) coordinates for a manuscript within the defined boundaries of a specified `region`.
    *   **Inputs:**
        *   `region` (`Region`): The enum value for the geographical region.
        *   `rng` (`np.random.Generator`): A seeded NumPy random number generator to ensure reproducibility.
    *   **Process:**
        *   Looks up the `x` and `y` bounds for the given `region` in `REGION_BOUNDS`.
        *   Uses `rng.uniform()` to sample a random floating-point `x` coordinate within `x_bounds` and a random `y` coordinate within `y_bounds`.
    *   **Output:** A tuple `(x, y)` representing the generated coordinates.
    *   **Error Handling:** Raises a `ValueError` if the provided `region` does not have defined bounds in `REGION_BOUNDS`.

This module is crucial for spatial modeling, enabling the simulation to factor in geographical proximity for processes like exemplar selection, while maintaining a clear separation of concerns regarding manuscript properties.

### `src/pasim/core/state.py`

This module defines the fundamental data structures and identity registries that represent the core state of the `pasim` simulation. It establishes the conceptual entities of `Material`, `Region`, `Script`, `Manuscript`, and `Witness`, along with registries to manage their instances.

**Key Components:**

*   **Enums:**
    *   **`Material`**: An `Enum` representing the physical material of a manuscript (e.g., `PARCHMENT`, `PAPYRUS`, `PAPER`).
    *   **`Region`**: An `Enum` representing geographical areas where manuscripts are located (e.g., `ASIA_MINOR`, `EGYPT`, `LEVANT`).
    *   **`Script`**: An `Enum` representing different script styles used in textual witnesses (e.g., `UNCIAL`, `MINUSCULE`).
*   **Dataclasses:**
    *   **`Manuscript`**: A dataclass representing a physical manuscript.
        *   `manuscript_id` (str): Unique identifier.
        *   `birth_tick` (int): Simulation tick when the manuscript is created.
        *   `death_tick` (int): Simulation tick when the manuscript ceases to exist.
        *   `material` (`Material`): Immutable material type.
        *   `region` (`Region`): Current geographical region, mutable over time due to migration.
        *   `location` (`Tuple[float, float]`): Precise (x,y) coordinates, mutable over time.
    *   **`Witness`**: A dataclass representing the textual content associated with a `Manuscript`.
        *   `witness_id` (str): Unique identifier for the textual content.
        *   `manuscript_id` (str): Foreign key linking to its parent `Manuscript`.
        *   `script` (`Script`): Immutable script style used.
        *   `metadata` (`Dict[str, Any]`): Optional field for arbitrary additional data.
*   **Registries:**
    *   **`ManuscriptRegistry`**: Manages a collection of all `Manuscript` objects.
        *   `add(manuscript)`: Adds a manuscript, enforcing `manuscript_id` uniqueness.
        *   `get(manuscript_id)`: Retrieves a manuscript by ID.
        *   Provides `__contains__`, `__len__`, and `items()` for collection management.
    *   **`WitnessRegistry`**: Manages a collection of all `Witness` objects.
        *   Requires a `ManuscriptRegistry` during initialization to validate `manuscript_id` references.
        *   `add(witness)`: Adds a witness, ensuring `witness_id` uniqueness and that its `manuscript_id` refers to an existing manuscript.
        *   `get(witness_id)`: Retrieves a witness by ID.
        *   Provides `__contains__`, `__len__`, and `items()` for collection management.
*   **`StateRegistry`**: A top-level dataclass that acts as a container for both `ManuscriptRegistry`, `WitnessRegistry`, and the `instance_texts` dictionary. It provides a centralized, consistent snapshot of all entities in the simulation without containing any simulation logic itself. The `WitnessRegistry` is initialized using the `ManuscriptRegistry` in `__post_init__`.

This module is foundational for defining the entities and their relationships within the simulation, ensuring data integrity and providing structured access to the simulation's current state.

### `src/pasim/core/tagged_string_constraints.py`

This module serves as the authoritative source for defining the legal state space of values within a "tagged string," which represents the textual content of a witness. It centralizes the fundamental constraints for textual segments, ensuring consistency and validity across all parts of the simulation that create or modify tagged strings, especially mutation operators and factory functions.

**Key Constants and Functions:**

*   **`LEGAL_SEGMENT_VALUES` (NDArray[np.int16]):** A NumPy array explicitly listing all permissible integer values that a single segment within a tagged string can hold (currently `[1, 2, 3, 4, 5]`). This acts as the "alphabet" for the textual model.
*   **`SegmentValue` (Type Alias):** An alias for `np.int16` for clarity in type hinting.
*   **`is_valid_segment_value(value: SegmentValue) -> bool`:**
    *   A simple predicate function that checks if a given integer `value` is present within `LEGAL_SEGMENT_VALUES`.
*   **`validate_tagged_string(tagged_string: NDArray[np.int16], config: SimulationConfig) -> None`:**
    *   A comprehensive validation function for an entire `tagged_string` (NumPy array).
    *   Performs multiple checks to ensure the string's integrity:
        *   Confirms it's a NumPy array.
        *   Verifies its length matches `config.text_length`.
        *   Checks that its data type is integer-like.
        *   Ensures that *all* values within the array are `is_valid_segment_value`.
    *   Raises `TypeError` or `ValueError` if any validation fails. Intended as a safeguard after creation or modification.
*   **`sample_alternative_value(current_value: SegmentValue, rng: np.random.Generator) -> SegmentValue`:**
    *   A critical utility for mutation logic. It samples a *new*, legal segment value that is explicitly *different* from the `current_value`.
    *   It achieves this by filtering `LEGAL_SEGMENT_VALUES` to exclude `current_value` and then randomly choosing from the remaining alternatives using the provided `rng` (ensuring deterministic selection).
    *   Raises `ValueError` if `current_value` itself is not valid.

This module is fundamental for maintaining the integrity and consistency of the textual data throughout the simulation, providing a robust foundation for modeling textual variation.

### `src/pasim/core/text_initialisation.py`

This module provides utility functions for generating the initial textual content (tagged strings) for new witness instances.

**Key Functionality:**

*   **`make_initial_text(config)`**: Creates a base tagged string for the autograph (the first witness instance). This function initializes a NumPy array of a specified `text_length` (from the simulation configuration) with a default legal segment value.

### `src/pasim/core/transmission.py`

This module defines the base rules for "base transmission" of textual content from one or more parent exemplars to a new copy. It focuses purely on the structural combination of source texts, deliberately excluding any considerations of scribal errors or mutations. This separation ensures a clean, deterministic base layer upon which more complex error models can be applied.

**Key Concepts:**

*   **Separation of Concerns:** Distinguishes the process of combining source texts from the process of introducing errors. This allows for isolated testing and clearer understanding of each stage.
*   **Determinism:** All functions in this module are deterministic given their inputs, especially when a seeded `rng` is used for tie-breaking.

**Key Functions:**

*   **`copy_from_single_exemplar(parent_text: TaggedString) -> TaggedString`:**
    *   **Purpose:** To simulate the most straightforward copying scenario: a perfect replication from a single source.
    *   **Functionality:** Takes a `TaggedString` (NumPy array) as `parent_text` and returns a new, independent deep copy of it. This ensures that subsequent modifications (e.g., mutations) to the copy do not affect the original.
    *   **Determinism:** Fully deterministic.
*   **`majority_from_exemplars(parent_texts: List[TaggedString], rng: np.random.Generator) -> TaggedString`:**
    *   **Purpose:** To combine textual content from multiple parent exemplars using a segment-wise majority voting mechanism. This models "contamination" or influence from multiple sources.
    *   **Inputs:**
        *   `parent_texts`: A list of `TaggedString` (NumPy arrays), representing the texts from multiple exemplars.
        *   `rng`: A NumPy random number generator, used *only* for deterministically breaking ties when multiple values share the highest frequency in a segment.
    *   **Functionality:**
        *   Validates that at least two parent texts are provided and that all parent texts have consistent shapes and data types.
        *   For each segment position (column) across all `parent_texts`:
            *   It counts the occurrences of each unique value.
            *   If there's a clear majority (single most common value), that value is selected for the new text.
            *   If there's a tie for the most common value, `rng.choice()` is used to randomly (but deterministically, given `rng`'s state) select one of the tied values.
        *   Constructs and returns a new `TaggedString` (NumPy array) containing the result of this majority voting.
    *   **Determinism:** Deterministic given the inputs and `rng` state.

This module is a foundational piece of the simulation, providing the mechanisms for how textual readings propagate and combine across generations of manuscripts before any scribal errors are introduced.

### `src/pasim/execution/__init__.py`

This file is an empty `__init__.py` file, serving primarily to mark the `execution` directory as a Python subpackage within `pasim`. It does not contain any functional code or exposed modules.

### `src/pasim/execution/batch.py`

This file is currently empty. It is intended for managing and running multiple simulations in batches, as described in `project_overview.md`. This would typically involve iterating through a set of configurations or seeds, orchestrating individual simulation runs, and potentially collecting their results.

### `src/pasim/execution/parallel.py`

This module provides the `run_parallel` function, which orchestrates multiple independent simulation runs concurrently using process-based parallelism. It manages unique seed generation for each run via `RNGContext`, implements robust retry logic for individual run failures, and aggregates results (including failure records) into a comprehensive summary. It is primarily called by `run_experiment` in the `orchestrator` module.

### `src/pasim/execution/runner.py`

This module provides the `run_single` function, the primary entry point for executing a single, in-memory simulation run. It handles loading and validating parameters from a YAML file, initializing the simulation state with a deterministic RNG, running the simulation, and returning a structured `SimulationResult` object containing the final state, genealogy graph, and telemetry data.

### `src/pasim/io/__init__.py`

This file is an empty `__init__.py` file, serving primarily to mark the `io` directory as a Python subpackage within `pasim`. It does not contain any functional code or exposed modules.

### `src/pasim/io/aggregation.py`

This file is currently empty. It is intended for aggregating results from multiple simulation runs, as described in `project_overview.md`. This would typically involve combining data from individual runs to produce summary statistics, averages, or other aggregated views.

### `src/pasim/io/formats.py`

This file is currently empty. It is intended for handling different data formats used for input or output in the simulation, as described in `project_overview.md`. This could include defining parsers for configuration files, serializers for simulation results, or adapters for various data storage types.

### `src/pasim/io/output.py`

This file is currently empty. It is intended for writing simulation results to various output destinations (e.g., files, databases), as described in `project_overview.md`. This would encompass functions for saving generated genealogies, textual data, and other simulation outputs.

### `src/pasim/utils/__init__.py`

This file is an empty `__init__.py` file, serving primarily to mark the `utils` directory as a Python subpackage within `pasim`. It does not contain any functional code or exposed modules.

### `src/pasim/utils/logging.py`

This file is currently empty. It is intended for implementing logging mechanisms within the simulation framework, as described in `project_overview.md`. This would typically involve configuring loggers, handlers, and formatters to capture and output various levels of information, warnings, and errors during simulation execution.

### `src/pasim/utils/timing.py`



This file is currently empty. It is intended for performance measurement and timing utilities within the simulation framework, as described in `project_overview.md`. This would typically involve functions or decorators for measuring execution time of various simulation components.
