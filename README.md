# pasim: Pericope Adulterae Simulation

`pasim` is a scientific simulation framework designed to model the historical transmission and evolution of the *Pericope Adulterae* (PA). It simulates manuscript genealogies, regional migration, historical shocks (like persecutions), and the gradual transition of writing materials and script styles.


## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/normansimonr/pericope_adulterae_simulation.git
    cd pericope_adulterae_simulation
    ```

2.  **Install dependencies**:
    ```bash
    poetry install
    ```

3.  **Activate the virtual environment** (optional):
    ```bash
    poetry shell
    ```

## Initialisation

To ensure code quality and consistent formatting, it is highly recommended to set up the pre-commit hooks:

```bash
poetry run pre-commit install
```

This will run linting (Ruff), type checking (Mypy), and tests (Pytest) before every commit.

## Running Experiments

The simulation is driven by YAML configuration files located in the `experiments/` directory.

### 1. List available experiments
To see all defined experiments and their current execution status:
```bash
poetry run pasim list
```

### 2. Run a simulation
To execute a specific experiment (e.g., the baseline):
```bash
poetry run pasim run experiments/exp001_baseline/params.yaml
```

**Persistence Levels**:
- `--persistence-level minimal` (Default): Only saves aggregated results (`results.csv`) and experiment metadata. Recommended for large sweeps.
- `--persistence-level full`: Saves everything, including full genealogies, textual states for every witness, and detailed logs for every run.

Example with full persistence:
```bash
poetry run pasim run experiments/exp001_baseline/params.yaml --persistence-level full
```

**Note: We recommend using full persistence in small batches only. If run in large batches, the artifacts may fill up your hard disk and cause your operating system to collapse.**


### 3. Inspect results
- **Aggregated Results**: Found in the experiment root, e.g., `experiments/exp001_baseline/results.csv`.
- **Detailed Artefacts**: (If using `full` persistence) Found in `experiments/exp001_baseline/runs/run_N/`.

### 4. Reset experiment data
To clean up all generated run data and metadata:
```bash
poetry run pasim reset
```

## Development and Testing

- **Run tests**:
  ```bash
  poetry run pasim tests
  # OR
  poetry run pytest
  ```
- **Profile performance**:
  ```bash
  poetry run python scripts/profile_run.py
  ```
