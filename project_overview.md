# Project Overview: Pericope Adulterae Simulation (pasim)

This document provides an overview of the `pasim` project, outlining its purpose, architectural structure, and general functioning.

## Purpose

The primary goal of this project is to simulate the textual transmission of the New Testament passage known as the Pericope Adulterae (John 7:53-8:11). It provides a framework for researchers to conduct computational simulations, specifically Monte Carlo simulations, of the copying process of this Greek passage over centuries. This allows for the study of how textual variations might have emerged and propagated through historical manuscript traditions.

## Core Concepts

To accurately model textual transmission, the simulation distinguishes between several key entities:

-   **Manuscript**: Represents a physical object (e.g., a codex, scroll) with tangible attributes such as its `birth_tick`, `death_tick`, `material` (parchment, papyrus, paper), `region`, and `location`. These attributes belong exclusively to the physical manuscript. Manuscripts are tracked in a `ManuscriptRegistry`.

-   **Witness**: Represents the textual content found within a specific manuscript. A Manuscript owns exactly one Witness. A Witness has its own unique ID and is linked to its parent Manuscript. Witnesses are tracked in a `WitnessRegistry`.

-   **Witness Instance / Genealogy Node**: This is an abstract representation in the genealogy graph. Each Witness has exactly one Witness Instance, which corresponds to a node in the `networkx` graph. This node stores only structural information (like its ID and links to its Witness and Manuscript via foreign keys) and its `birth_tick`. Crucially, **it does NOT store regional, geographic, material, or `death_tick` data**, as these belong to the `Manuscript` object. The graph thus focuses purely on the lineage and copying relationships between textual instances.

The relationship can be visualized as:
`Manuscript` (physical attributes)
  `|`
  `+-- owns --> Witness` (textual content)
           `|`
           `+-- represented by --> Witness Instance` (graph node)

This separation ensures that the abstract genealogical structure remains clean and unburdened by the evolving physical properties of the manuscripts.

## Architecture and Structure

The project `pasim` is designed as a modular framework, providing a structured approach to building and running scientific simulations. The core components are organized within the `src/pasim` directory, with distinct responsibilities:

-   **`core/`**: This module is intended to house the core, pure, and deterministic simulation logic. This includes the fundamental rules and processes governing the textual transmission model, independent of I/O or execution concerns.
    -   `scribal_rules.py`: Implements the composite scribal rule, which is the high-level pipeline for generating a new textual witness. It composes the base transmission, reputation-to-error-intensity, and mutation modules to simulate the full act of copying.
    -   `genealogy.py`: Defines the structural genealogy of witness instances using a `networkx` directed acyclic graph (DAG). It tracks ancestry, timing, and topology (who copied from whom and when).
    -   `genealogy_generator.py`: Orchestrates the deterministic, tick-based generation of the genealogy graph. It uses a state-machine-like process, advancing tick by tick and hosting the injection of scientific rules. It now manages the lifecycle of manuscripts, including their deterministic death at a scheduled tick and **demand-based spawning of new manuscripts** to meet exogenous regional requirements.
    -   `tagged_string_constraints.py`: Defines the legal "alphabet" for segment values in a tagged string. It provides a single, authoritative source of truth for the valid state space, ensuring all mutation and factory functions operate within explicitly enforced boundaries.
    -   `transmission.py`: Defines the base rules for how a new tagged string is generated from parent exemplars (e.g., majority voting). This represents the ideal, error-free copying process before any scribal mutations are applied.
    -   `reputation.py`: Defines the policy layer that maps a witness's reputation level to an expected proportion of segments that will mutate during copying. This translates the abstract concept of reputation into a concrete error intensity.
    -   `mutation.py`: Defines the mechanical operator that applies scribal mutations to a tagged string. It takes an expected proportion of segments to change and, under controlled randomness, alters the required number of segments to other legal values.
    -   `model.py`: Defines the core data structure for the textual state (the "tagged string") and provides factory functions to create, copy, and mutate it. A tagged string is a NumPy array of integers representing the sequence of readings (textual variants) in a witness instance.
    -   `rng.py`: Implements the centralized random number generation (RNG) factory. This factory ensures full reproducibility across runs and safe parallel execution by managing `numpy.random.Generator` and `numpy.random.SeedSequence` objects, derived from a single batch-level seed.
    -   `spatial.py`: Provides utilities for handling spatial assignments of manuscripts, ensuring region-specific coordinate generation and maintaining the architectural separation of spatial properties from genealogy nodes.
    -   `state.py`: Defines the core data structures and identity registries for the simulation. This includes `Manuscript` and `Witness` data classes, as well as the `ManuscriptRegistry` and `WitnessRegistry` that manage the unique identities of these entities. A top-level `StateRegistry` class holds these registries, providing a consistent snapshot of what exists in the simulation.
-   **`config/`**: Manages the configuration and parameters for the simulations.
    -   `schema.py`: Will define the validation schema for simulation configuration files.
-   **`execution/`**: Handles the orchestration and running of simulations.
    -   `runner.py`: Will execute a single simulation run.
    -   `batch.py`: For managing and running multiple simulations in batches.
    -   `parallel.py`: For parallelizing simulation execution.
-   **`analysis/`**: Provides tools for processing, analyzing, and visualizing simulation results.
    -   `metrics.py`: For defining and calculating simulation metrics.
    -   `plots.py`: For generating visualizations of results.
    -   `statistics.py`: For statistical analysis of simulation outputs.
-   **`io/`**: Manages input and output operations, including reading initial data and writing simulation results.
    -   `formats.py`: For handling different data formats.
    -   `output.py`: For writing simulation outputs.
    -   `aggregation.py`: For aggregating results from multiple runs.
-   **`utils/`**: Contains common utility functions used across the project.
    -   `logging.py`: For logging mechanisms.
    -   `timing.py`: For performance measurement.
-   **`cli.py`**: The command-line interface entry point for interacting with the simulation framework.

## General Functioning (Planned)

The `pasim` framework will facilitate the following general workflow for researchers:

1.  **Configuration**: Users will define simulation parameters (e.g., number of scribes, error rates, manuscript branching, simulation duration, **regional demand for new manuscripts**) using configuration files, likely validated by the `config/schema.py`.
2.  **Initialization**: The simulation will initialize its state based on an initial text (the Pericope Adulterae) and the specified configuration.
3.  **Execution**: The `execution/` module will manage the running of Monte Carlo simulations. This involves iteratively applying copying rules, stochastic variations (defined in `core/model.py` and utilizing `core/rng.py`), and **dynamically spawning new manuscripts based on regional demand** to the textual state over a simulated historical period.
4.  **Data Collection**: Throughout the simulation, relevant data points (e.g., changes in text, manuscript lineages, manuscript population dynamics) will be collected and stored via the `io/` module.
5.  **Analysis**: After simulation completion, the `analysis/` module will be used to process the collected data, calculate metrics, generate plots, and perform statistical analyses to derive insights into textual transmission.
6.  **Reporting**: Results and analyses will be outputted in various formats for researchers to study.

This setup enables researchers to explore various hypotheses about the textual history of the Pericope Adulterae by adjusting simulation parameters and observing the emergent textual traditions.
