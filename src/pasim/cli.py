import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

import pasim.execution.orchestrator

logger = logging.getLogger("pasim")


# --- Global Settings ---
EXPERIMENTS_DIR = Path("experiments")
DOCS_EXPERIMENTS_PATH = Path("docs/experiments.md")
PARAMS_TEMPLATE_PATH = EXPERIMENTS_DIR / "params_template.yaml"

# --- CLI Utility Functions ---


def _configure_logging(verbose: bool) -> None:
    """Configures the logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    # Clear existing handlers to prevent duplicate output
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)  # Change to sys.stderr
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    if verbose:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _load_params_file(params_path: Path) -> Dict[str, Any]:
    """Loads and returns parameters from a YAML file."""
    if not params_path.is_file():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError(f"Invalid YAML in {params_path}: Expected a dictionary.")
    return params


def _validate_params_path(params_path_str: str) -> Path:
    """Validates if the given path is a valid params.yaml file."""
    params_path = Path(params_path_str)
    if not params_path.is_file():
        logger.error(f"Error: Parameters file not found at '{params_path}'.")
        sys.exit(1)
    if params_path.name != "params.yaml":
        logger.warning(f"Warning: The parameters file is conventionally named 'params.yaml', but found '{params_path.name}'.")
    return params_path


# --- Commands ---


def _cmd_help(args: argparse.Namespace) -> None:
    """Displays help information for the CLI."""
    print("--- pasim CLI Usage ---")
    print("\nAvailable Commands:")
    print("  help                  - Displays this help message.")
    print("  run <params_path>     - Runs a simulation experiment.")
    print("  reset [--force]       - Cleans up experiment run data.")
    print("  tests                 - Runs the project's tests.")
    print("  list                  - Lists all experiments and their statuses.")
    print("\nGlobal Flags:")
    print("  --verbose, -v         - Enable verbose output for more detailed logs.")
    print("\nExperiment Directory Structure:")
    print("Each experiment should be in its own directory, e.g., 'experiments/my_experiment/'.")
    print("Inside, there should be a 'params.yaml' file configuring the experiment.")
    print("Simulation results are stored in auto-generated 'runs/' subdirectories.")
    print(f"\nFor more detailed documentation, see {DOCS_EXPERIMENTS_PATH}.")
    print("\nTo run an experiment, use 'pasim run <path/to/params.yaml>'.")
    print("Example: pasim run experiments/exp001_baseline/params.yaml")


def _check_params_against_template(params_path: Path) -> None:
    """Checks the given parameters file against the canonical template for mismatches."""
    if not PARAMS_TEMPLATE_PATH.is_file():
        logger.warning(f"Warning: Canonical template not found at {PARAMS_TEMPLATE_PATH}. Skipping structure check.")
        return

    try:
        user_params = _load_params_file(params_path)
        template_params = _load_params_file(PARAMS_TEMPLATE_PATH)
    except Exception as e:
        logger.warning(f"Warning: Could not perform parameter template check: {e}")
        return

    user_keys = set(user_params.keys())
    template_keys = set(template_params.keys())

    missing_in_user = template_keys - user_keys
    extra_in_user = user_keys - template_keys

    if missing_in_user:
        error_msg = (
            f"Validation Error: The following parameters are present in the template but missing in your file: "
            f"{sorted(list(missing_in_user))}\nRefer to experiments/params_template.yaml for the recommended configuration structure."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if extra_in_user:
        logger.warning(f"Warning: Your parameters file contains keys not present in the template: {sorted(list(extra_in_user))}")


def _cmd_run(args: argparse.Namespace) -> None:
    """Runs a simulation experiment."""
    _configure_logging(args.verbose)
    params_path = _validate_params_path(args.params_path)

    try:
        # Perform structural check against template before execution
        _check_params_against_template(params_path)

        logger.info(f"Starting experiment from {params_path}...")
        summary = pasim.execution.orchestrator.run_experiment(params_path, persistence_level=args.persistence_level)

        failed_runs = summary.get("failed_runs", 0)
        total_runs = summary.get("total_runs_attempted") or summary.get("total_requested_runs", "unknown")

        if failed_runs > 0:
            logger.error(f"Experiment completed with {failed_runs} failures out of {total_runs} runs.")
            if args.verbose:
                for record in summary.get("failure_records", []):
                    logger.error(f"  Run (seed: {record['seed']}) failed: {record['error']} (attempt: {record['attempt']})")
            sys.exit(1)
        else:
            logger.info(f"Experiment completed successfully: {summary.get('successful_runs', 0)} runs completed.")
            sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during experiment execution: {e}", exc_info=args.verbose)
        sys.exit(1)


def _get_reset_targets() -> tuple[List[Path], List[Path]]:
    """Identifies the directories and files to be removed during a reset."""
    target_dirs: List[Path] = []
    target_files: List[Path] = []

    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if exp_dir.is_dir() and exp_dir.name != PARAMS_TEMPLATE_PATH.stem:
            runs_dir = exp_dir / "runs"
            if runs_dir.is_dir():
                target_dirs.append(runs_dir)

            metadata_file = exp_dir / "experiment_metadata.json"
            if metadata_file.is_file():
                target_files.append(metadata_file)
    return target_dirs, target_files


def _confirm_reset(target_dirs: List[Path], target_files: List[Path]) -> bool:
    """Asks the user for confirmation before resetting data."""
    print("The following experiment run data will be removed:")
    for d in target_dirs:
        print(f"  - {d}")
    for f in target_files:
        print(f"  - {f}")

    confirmation = input("Are you sure you want to proceed? (y/N): ")
    return confirmation.lower() == "y"


def _perform_reset(target_dirs: List[Path], target_files: List[Path]) -> None:
    """Performs the actual removal of experiment data."""
    for d in target_dirs:
        try:
            shutil.rmtree(d)
            logger.info(f"Removed: {d}")
        except Exception as e:
            logger.error(f"Failed to remove directory {d}: {e}")
            sys.exit(1)

    for f in target_files:
        try:
            os.remove(f)
            logger.info(f"Removed: {f}")
        except Exception as e:
            logger.error(f"Failed to remove file {f}: {e}")
            sys.exit(1)


def _cmd_reset(args: argparse.Namespace) -> None:
    """Cleans up experiment run data."""
    _configure_logging(args.verbose)

    target_dirs, target_files = _get_reset_targets()

    if not target_dirs and not target_files:
        logger.info("No experiment run data found to reset.")
        sys.exit(0)

    if not args.force and not _confirm_reset(target_dirs, target_files):
        logger.info("Reset operation cancelled by user.")
        sys.exit(0)

    _perform_reset(target_dirs, target_files)

    logger.info("Reset complete.")
    sys.exit(0)


def _cmd_tests(args: argparse.Namespace) -> None:
    """Runs the project's tests."""
    _configure_logging(args.verbose)
    try:
        import pytest
    except ImportError:
        logger.critical("pytest is not installed. Please install dev dependencies with 'poetry install --with dev'.")
        sys.exit(1)

    pytest_args = ["--exitfirst", "--disable-warnings"]
    if args.verbose:
        pytest_args.append("-v")

    # Run pytest and exit with its exit code
    logger.info("Running project tests...")
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


def _get_experiment_info(exp_dir: Path) -> Dict[str, Any]:
    """Retrieves status and run counts for a single experiment directory."""
    params_file = exp_dir / "params.yaml"
    metadata_file = exp_dir / "experiment_metadata.json"

    exp_name = exp_dir.name
    n_runs_requested = "N/A"
    n_runs_completed = "N/A"
    exp_status = "Pending / No Metadata"

    if params_file.is_file():
        try:
            params = _load_params_file(params_file)
            n_runs_requested = params.get("n_runs", "N/A")
        except Exception as e:
            logger.warning(f"Could not load params for {exp_name}: {e}")
            n_runs_requested = "Error"
    else:
        logger.warning(f"No params.yaml found for experiment: {exp_name}")

    if metadata_file.is_file():
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            n_runs_completed = metadata["run_counts"]["successful"] + metadata["run_counts"]["failed"]
            exp_status = metadata.get("execution_status", "UNKNOWN").replace("_", " ").title()

        except Exception as e:
            logger.warning(f"Could not load metadata for {exp_name}: {e}")
            exp_status = "Metadata Error"

    return {
        "name": exp_name,
        "requested": n_runs_requested,
        "completed": n_runs_completed,
        "status": exp_status,
    }


def _collect_experiments_data() -> List[Dict[str, Any]]:
    """Scans the experiments directory and collects information for all experiments."""
    experiments_data: List[Dict[str, Any]] = []

    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if exp_dir.is_dir() and exp_dir.name != PARAMS_TEMPLATE_PATH.stem:
            info = _get_experiment_info(exp_dir)
            experiments_data.append(info)
    return experiments_data


def _print_experiments_table(experiments_data: List[Dict[str, Any]]) -> None:
    """Prints the experiment data in a formatted table."""
    # Print header
    print(f"\n{'Experiment Name':<30} {'Requested Runs':<15} {'Completed Runs':<15} {'Status':<25}")
    print(f"{'-' * 30:<30} {'-' * 15:<15} {'-' * 15:<15} {'-' * 25:<25}")

    for exp in sorted(experiments_data, key=lambda x: x["name"]):
        print(f"{exp['name']:<30} {str(exp['requested']):<15} {str(exp['completed']):<15} {exp['status']:<25}")


def _cmd_list(args: argparse.Namespace) -> None:
    """Lists all experiments and their statuses."""
    _configure_logging(args.verbose)

    logger.info(f"Scanning experiments in '{EXPERIMENTS_DIR}'...")

    experiments_data = _collect_experiments_data()

    if not experiments_data:
        logger.info("No experiment directories found (excluding params_template).")
        sys.exit(0)

    _print_experiments_table(experiments_data)
    sys.exit(0)


# --- Main CLI Entrypoint ---


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point for the pasim CLI."""
    # Parent parser for global arguments like --verbose
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")

    parser = argparse.ArgumentParser(
        prog="pasim",
        description="CLI for the Pericope Adulterae Simulation (pasim) project.",
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[parent_parser],  # Inherit verbose flag
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Help command
    help_parser = subparsers.add_parser("help", help="Show this help message and exit.", parents=[parent_parser])
    help_parser.set_defaults(func=_cmd_help)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a simulation experiment.", parents=[parent_parser])
    run_parser.add_argument("params_path", type=str, help="Path to the experiment's params.yaml file.")
    run_parser.add_argument(
        "--persistence-level",
        type=str,
        choices=["minimal", "full"],
        default="minimal",
        help="Persistence level: 'minimal' (aggregated results only) or 'full' (all artefacts). Default: 'minimal'.",
    )
    run_parser.set_defaults(func=_cmd_run)

    # Reset command
    reset_parser = subparsers.add_parser(
        "reset", help="Clean up experiment run data (removes 'runs/' folders and 'experiment_metadata.json').", parents=[parent_parser]
    )
    reset_parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation prompt.")
    reset_parser.set_defaults(func=_cmd_reset)

    # Tests command
    tests_parser = subparsers.add_parser("tests", help="Run the project's pytest suite.", parents=[parent_parser])
    tests_parser.set_defaults(func=_cmd_tests)

    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments and their current status.", parents=[parent_parser])
    list_parser.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)

    # If the user just types `pasim` (no subcommand) or `pasim -h`, show the main help
    # This check is technically redundant with required=True in subparsers,
    # but good for explicit clarity if required=True were to change.
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # The --help/-h flag is now handled by argparse directly on the main parser
    # or on subparsers when parents=[parent_parser] is used.
    # No need for manual sys.argv check.

    args.func(args)


if __name__ == "__main__":
    main()
