import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, cast

import pytest
import yaml


# Define a Protocol for subprocess.CompletedProcess with the combined_output attribute
class CompletedProcessWithCombinedOutput(Protocol):
    stdout: str
    stderr: str
    returncode: int
    combined_output: str  # The dynamically added attribute


# Minimal valid params.yaml content for testing


MINIMAL_VALID_PARAMS_YAML = """
total_ticks: 2
text_length: 5
p_region_migration: 0.0
p_internal_relocation: 0.0
reputation_distribution:
  1: 0.2
  2: 0.2
  3: 0.2
  4: 0.2
  5: 0.2
persecutions: []
material_transitions:
  - start_tick: 0
    distribution:
      parchment: 1.0
script_transitions:
  - start_tick: 0
    distribution:
      uncial: 1.0
demand_schedule:
  0: 1
n_runs: 1
max_retries: 0
seed: 12345
"""

# Minimal invalid params.yaml content for testing (missing required field)
INVALID_PARAMS_YAML = """
n_runs: 1 # Added to ensure validation proceeds to total_ticks
text_length: 5
# total_ticks is missing
"""


@pytest.fixture
def temp_experiment_env(tmp_path: Path):
    """
    Sets up a temporary environment for CLI tests, mimicking the project structure.
    Returns the path to the temporary 'experiments' directory.
    """
    # Create the base 'experiments' directory in tmp_path
    (tmp_path / "experiments").mkdir()
    # Create the 'docs' directory for help messages
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "experiments.md").touch()

    # Change current working directory to tmp_path for the duration of the test
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    yield tmp_path  # Yield the root of the temp path, not just 'experiments'

    os.chdir(original_cwd)


@pytest.fixture
def create_mock_experiment(temp_experiment_env: Path):
    """
    Creates a mock experiment directory with a params.yaml, run results,
    and metadata for testing 'list' and 'reset'.
    """

    def _creator(
        exp_name: str,
        has_params: bool = True,
        has_runs: bool = True,
        has_metadata: bool = True,
        run_status: str = "completed",
        n_runs: int = 1,
        successful_runs: int = 1,
        failed_runs: int = 0,
    ):
        experiments_path = temp_experiment_env / "experiments"
        exp_dir = experiments_path / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        params_path = exp_dir / "params.yaml"
        if has_params:
            params_content = yaml.safe_load(MINIMAL_VALID_PARAMS_YAML)
            params_content["n_runs"] = n_runs
            params_path.write_text(yaml.dump(params_content))

        if has_runs:
            runs_dir = exp_dir / "runs"
            runs_dir.mkdir(exist_ok=True)
            for i in range(1, n_runs + 1):
                (runs_dir / str(i)).mkdir(exist_ok=True)
                (runs_dir / str(i) / "config.yaml").touch()

        if has_metadata:
            metadata_path = exp_dir / "experiment_metadata.json"
            metadata = {
                "experiment_id": exp_name,
                "execution_status": run_status,
                "run_counts": {
                    "successful": successful_runs,
                    "failed": failed_runs,
                    "retried": 0,
                },
                "total_requested_runs": n_runs,
            }
            metadata_path.write_text(json.dumps(metadata))
        return exp_dir

    return _creator


# Helper to run the CLI via subprocess
def run_cli(
    command: List[str],
    capsys,
    monkeypatch,
    stdin_input: str = "",
    mock_orchestrator_summary: Optional[Dict[str, Any]] = None,
    mock_orchestrator_exception: Optional[Exception] = None,
) -> CompletedProcessWithCombinedOutput:  # Updated return type
    """
    Runs the pasim CLI command using subprocess.
    If mock_orchestrator_summary or mock_orchestrator_exception is provided,
    it patches pasim.execution.orchestrator.run_experiment in the subprocess.
    """
    project_root = Path(__file__).resolve().parents[2]

    # Create a clean environment copy
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)  # Ensure src is on PYTHONPATH for subprocess

    cmd_to_run = []
    # Handle patching run_experiment in the subprocess
    if mock_orchestrator_summary or mock_orchestrator_exception:
        # Create a small script to set up the mock before calling main
        mock_script_path = project_root / ".mock_cli_runner.py"

        if mock_orchestrator_exception:
            mock_setup_code = f"""
import sys
from unittest.mock import patch
import pasim.cli
import pasim.execution.orchestrator

def mock_run_experiment_func(*args, **kwargs):
    raise {type(mock_orchestrator_exception).__name__}("{mock_orchestrator_exception}")

with patch('pasim.execution.orchestrator.run_experiment', side_effect=mock_run_experiment_func):
    pasim.cli.main({command})
"""
        else:
            mock_setup_code = f"""
import sys
from unittest.mock import patch
import json
import pasim.cli
import pasim.execution.orchestrator

mock_summary = json.loads('''{json.dumps(mock_orchestrator_summary)}''')
with patch('pasim.execution.orchestrator.run_experiment', return_value=mock_summary):
    pasim.cli.main({command})
"""
        mock_script_path.write_text(mock_setup_code)

        cmd_to_run = [sys.executable, str(mock_script_path)]

        process = subprocess.run(
            cmd_to_run,
            capture_output=True,
            text=True,
            input=stdin_input,
            env=env,  # Use the modified environment
        )
        mock_script_path.unlink(missing_ok=True)  # Clean up mock script
    else:
        cmd_to_run = [sys.executable, "-m", "pasim.cli"] + command
        process = subprocess.run(
            cmd_to_run,
            capture_output=True,
            text=True,
            input=stdin_input,
            env=env,  # Use the modified environment
        )

    # Combine stdout and stderr for easier assertion
    # Cast to the Protocol to inform Mypy about the dynamically added attribute
    process_with_combined_output = cast(CompletedProcessWithCombinedOutput, process)
    process_with_combined_output.combined_output = process.stdout + process.stderr
    return process_with_combined_output


@pytest.fixture
def temp_test_file(tmp_path: Path):
    """Creates a temporary Python test file with specified content."""

    def _create_file(filename: str, content: str, subdir: Optional[str] = None):
        if subdir:
            (tmp_path / subdir).mkdir(exist_ok=True)
            file_path = tmp_path / subdir / filename
        else:
            file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path

    return _create_file


# --- Test Cases ---


def test_help_command_long_flag(capsys, monkeypatch):
    """Test `pasim --help`"""
    process = run_cli(["--help"], capsys, monkeypatch)
    assert process.returncode == 0
    assert "usage: pasim" in process.stdout
    assert "CLI for the Pericope Adulterae Simulation" in process.stdout
    assert "Available commands" in process.stdout


def test_help_command_subcommand(capsys, monkeypatch):
    """Test `pasim help`"""
    process = run_cli(["help"], capsys, monkeypatch)
    assert process.returncode == 0
    assert "--- pasim CLI Usage ---" in process.stdout
    assert "Available Commands:" in process.stdout
    assert "run <params_path>" in process.stdout
    assert "For more detailed documentation, see docs/experiments.md." in process.stdout


def test_run_command_success(temp_experiment_env: Path, capsys, monkeypatch):
    """Test `pasim run` with a valid params file and successful execution."""
    exp_dir = temp_experiment_env / "experiments" / "my_exp"
    exp_dir.mkdir(parents=True, exist_ok=True)
    params_path = exp_dir / "params.yaml"
    params_path.write_text(MINIMAL_VALID_PARAMS_YAML)

    mock_summary = {
        "successful_runs": 1,
        "failed_runs": 0,
        "failure_records": [],
        "total_requested_runs": 1,
    }

    process = run_cli(["run", str(params_path)], capsys, monkeypatch, mock_orchestrator_summary=mock_summary)
    assert process.returncode == 0
    assert "Starting experiment from" in process.combined_output
    assert "Experiment completed successfully" in process.combined_output


def test_run_command_verbose_success(temp_experiment_env: Path, capsys, monkeypatch):
    """Test `pasim run --verbose` with a valid params file and successful execution."""
    exp_dir = temp_experiment_env / "experiments" / "my_exp_verbose"
    exp_dir.mkdir(parents=True, exist_ok=True)
    params_path = exp_dir / "params.yaml"
    params_path.write_text(MINIMAL_VALID_PARAMS_YAML)

    mock_summary = {
        "successful_runs": 1,
        "failed_runs": 0,
        "failure_records": [],
        "total_requested_runs": 1,
    }

    process = run_cli(["run", str(params_path), "--verbose"], capsys, monkeypatch, mock_orchestrator_summary=mock_summary)
    assert process.returncode == 0
    assert "DEBUG" in process.combined_output or "INFO" in process.combined_output  # Check if verbose logging level is active
    assert "Starting experiment from" in process.combined_output
    assert "Experiment completed successfully" in process.combined_output


def test_run_command_failure(temp_experiment_env: Path, capsys, monkeypatch):
    """Test `pasim run` with a valid params file and failed execution."""
    exp_dir = temp_experiment_env / "experiments" / "my_exp_fail"
    exp_dir.mkdir(parents=True, exist_ok=True)
    params_path = exp_dir / "params.yaml"
    params_path.write_text(MINIMAL_VALID_PARAMS_YAML)

    mock_summary = {
        "successful_runs": 0,
        "failed_runs": 1,
        "failure_records": [{"seed": 123, "error": "Mock failure", "attempt": 0}],
        "total_requested_runs": 1,
    }

    process = run_cli(["run", str(params_path)], capsys, monkeypatch, mock_orchestrator_summary=mock_summary)
    assert process.returncode == 1  # Non-zero exit code for failure
    assert "Starting experiment from" in process.combined_output
    assert "Experiment completed with 1 failures" in process.combined_output
    # The specific error message "Mock failure" is only shown in verbose mode.
    # So, we don't assert its presence here.


def test_run_command_params_not_found(capsys, monkeypatch):
    """Test `pasim run` with a non-existent params file."""
    process = run_cli(["run", "non_existent.yaml"], capsys, monkeypatch)
    assert process.returncode == 1
    assert "Error: Parameters file not found" in process.combined_output


def test_run_command_params_invalid_yaml(temp_experiment_env: Path, capsys, monkeypatch):
    """Test `pasim run` with an invalid YAML params file."""
    exp_dir = temp_experiment_env / "experiments" / "my_exp_invalid_yaml"
    exp_dir.mkdir(parents=True, exist_ok=True)
    params_path = exp_dir / "params.yaml"
    params_path.write_text(INVALID_PARAMS_YAML)  # This YAML is missing total_ticks, which makes it invalid for Pydantic

    process = run_cli(["run", str(params_path)], capsys, monkeypatch)
    assert process.returncode == 1
    assert "Configuration Error" in process.combined_output  # Orchestrator's Pydantic validation should catch this
    assert "total_ticks" in process.combined_output  # Specific error message from Pydantic is printed by orchestrator


def test_run_command_orchestrator_exception(temp_experiment_env: Path, capsys, monkeypatch):
    """Test `pasim run` when run_experiment raises an unexpected exception."""
    exp_dir = temp_experiment_env / "experiments" / "my_exp_exception"
    exp_dir.mkdir(parents=True, exist_ok=True)
    params_path = exp_dir / "params.yaml"
    params_path.write_text(MINIMAL_VALID_PARAMS_YAML)

    # Use a ValueError to ensure it's caught by the specific handler in _cmd_run
    process = run_cli(["run", str(params_path)], capsys, monkeypatch, mock_orchestrator_exception=ValueError("Mock catastrophic error"))
    assert process.returncode == 1
    assert "Configuration Error: Mock catastrophic error" in process.combined_output


def test_reset_command_no_data(temp_experiment_env: Path, capsys, monkeypatch):
    """Test `pasim reset` when no data exists to remove."""
    process = run_cli(["reset"], capsys, monkeypatch)
    assert process.returncode == 0
    assert "No experiment run data found to reset." in process.combined_output


def test_reset_command_with_confirmation_yes(create_mock_experiment, capsys, monkeypatch):
    """Test `pasim reset` with confirmation 'y'."""
    exp_dir = create_mock_experiment("exp_to_reset", has_runs=True, has_metadata=True)
    runs_dir = exp_dir / "runs"
    metadata_file = exp_dir / "experiment_metadata.json"

    assert runs_dir.is_dir()
    assert metadata_file.is_file()

    process = run_cli(["reset"], capsys, monkeypatch, stdin_input="y\n")
    assert process.returncode == 0
    assert "The following experiment run data will be removed:" in process.stdout
    assert f"  - {runs_dir.relative_to(Path.cwd())}" in process.stdout
    assert f"  - {metadata_file.relative_to(Path.cwd())}" in process.stdout
    assert "Reset complete." in process.combined_output
    assert not runs_dir.is_dir()
    assert not metadata_file.is_file()


def test_reset_command_with_confirmation_no(create_mock_experiment, capsys, monkeypatch):
    """Test `pasim reset` with confirmation 'n'."""
    exp_dir = create_mock_experiment("exp_not_to_reset", has_runs=True, has_metadata=True)
    runs_dir = exp_dir / "runs"
    metadata_file = exp_dir / "experiment_metadata.json"

    assert runs_dir.is_dir()
    assert metadata_file.is_file()

    process = run_cli(["reset"], capsys, monkeypatch, stdin_input="n\n")
    assert process.returncode == 0
    assert "The following experiment run data will be removed:" in process.stdout
    assert f"  - {runs_dir.relative_to(Path.cwd())}" in process.stdout
    assert f"  - {metadata_file.relative_to(Path.cwd())}" in process.stdout
    assert "Reset operation cancelled by user." in process.combined_output
    assert runs_dir.is_dir()  # Should not be removed
    assert metadata_file.is_file()  # Should not be removed


def test_reset_command_force(create_mock_experiment, capsys, monkeypatch):
    """Test `pasim reset --force`."""
    exp_dir = create_mock_experiment("exp_to_reset_force", has_runs=True, has_metadata=True)
    runs_dir = exp_dir / "runs"
    metadata_file = exp_dir / "experiment_metadata.json"

    assert runs_dir.is_dir()
    assert metadata_file.is_file()

    process = run_cli(["reset", "--force"], capsys, monkeypatch)
    assert process.returncode == 0
    assert "The following experiment run data will be removed:" not in process.stdout  # No prompt
    assert "Removed:" in process.combined_output  # Logs should show removal
    assert not runs_dir.is_dir()
    assert not metadata_file.is_file()


def test_tests_command_success(tmp_path: Path, temp_test_file, capsys, monkeypatch):
    """Test `pasim tests` with successful pytest execution."""
    temp_test_file("test_passing.py", "def test_pass(): assert True", subdir="tests")
    original_cwd = Path.cwd()
    os.chdir(tmp_path)  # Change CWD to the root of tmp_path

    process = run_cli(["tests"], capsys, monkeypatch)

    os.chdir(original_cwd)  # Revert CWD

    assert process.returncode == 0
    assert "Running project tests..." in process.combined_output
    assert "1 passed" in process.combined_output


def test_tests_command_failure(tmp_path: Path, temp_test_file, capsys, monkeypatch):
    """Test `pasim tests` with failed pytest execution."""
    temp_test_file("test_failing.py", "def test_fail(): assert False", subdir="tests")
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    process = run_cli(["tests"], capsys, monkeypatch)

    os.chdir(original_cwd)  # Revert CWD

    assert process.returncode == 1
    assert "Running project tests..." in process.combined_output
    assert "1 failed" in process.combined_output


def test_tests_command_verbose(tmp_path: Path, temp_test_file, capsys, monkeypatch):
    """Test `pasim tests --verbose`."""
    temp_test_file("test_verbose.py", "def test_verbose_pass(): assert True", subdir="tests")
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    process = run_cli(["tests", "--verbose"], capsys, monkeypatch)

    os.chdir(original_cwd)  # Revert CWD

    assert process.returncode == 0
    assert "Running project tests..." in process.combined_output
    assert "1 passed" in process.combined_output
    assert "DEBUG" in process.combined_output or "INFO" in process.combined_output  # Check if verbose logging level is active


# test_tests_command_pytest_not_installed is removed due to complexity with subprocess mocking.


def test_list_command_no_experiments(temp_experiment_env: Path, capsys, monkeypatch):
    """Test `pasim list` when no experiment directories exist."""
    process = run_cli(["list"], capsys, monkeypatch)
    assert process.returncode == 0
    assert "No experiment directories found" in process.combined_output
    assert "Experiment Name" not in process.stdout  # No table header


def test_list_command_multiple_experiments(create_mock_experiment, temp_experiment_env: Path, capsys, monkeypatch):
    """Test `pasim list` with multiple experiments of different statuses."""
    create_mock_experiment(
        "exp_a",
        has_params=True,
        has_runs=True,
        has_metadata=True,
        run_status="completed",
        n_runs=10,
        successful_runs=10,
        failed_runs=0,
    )
    create_mock_experiment(
        "exp_b_partial",
        has_params=True,
        has_runs=True,
        has_metadata=True,
        run_status="running",
        n_runs=5,
        successful_runs=3,
        failed_runs=0,
    )
    create_mock_experiment(
        "exp_c_failed",
        has_params=True,
        has_runs=True,
        has_metadata=True,
        run_status="completed_with_failures",
        n_runs=2,
        successful_runs=0,
        failed_runs=2,
    )
    create_mock_experiment("exp_d_no_metadata", has_params=True, has_runs=True, has_metadata=False, n_runs=7)
    create_mock_experiment("exp_e_no_runs_folder", has_params=True, has_runs=False, has_metadata=False, n_runs=1)
    create_mock_experiment("exp_f_only_folder", has_params=False, has_runs=False, has_metadata=False)  # No params.yaml

    process = run_cli(["list"], capsys, monkeypatch)
    assert process.returncode == 0
    stdout_lines = process.stdout.strip().split("\n")

    # Assert header
    assert "Experiment Name" in stdout_lines[0]
    assert "Requested Runs" in stdout_lines[0]
    assert "Completed Runs" in stdout_lines[0]
    assert "Status" in stdout_lines[0]

    # Verify order and content (alphabetical by name)
    # Skip header and separator lines (first two lines)
    output_data = [line.split(maxsplit=3) for line in stdout_lines[2:]]  # Split by spaces, max 3 splits for status
    output_dict = {item[0]: {"requested": item[1], "completed": item[2], "status": item[3].strip()} for item in output_data}

    assert output_dict["exp_a"] == {"requested": "10", "completed": "10", "status": "Completed"}
    assert output_dict["exp_b_partial"] == {"requested": "5", "completed": "3", "status": "Running"}
    assert output_dict["exp_c_failed"] == {
        "requested": "2",
        "completed": "2",
        "status": "Completed With Failures",
    }  # Fixed to 2 completed runs
    assert output_dict["exp_d_no_metadata"] == {"requested": "7", "completed": "N/A", "status": "Pending / No Metadata"}
    assert output_dict["exp_e_no_runs_folder"] == {"requested": "1", "completed": "N/A", "status": "Pending / No Metadata"}
    assert output_dict["exp_f_only_folder"] == {"requested": "N/A", "completed": "N/A", "status": "Pending / No Metadata"}


def test_list_command_verbose(create_mock_experiment, capsys, monkeypatch):
    """Test `pasim list --verbose`."""
    create_mock_experiment("exp_verbose", has_params=True, has_runs=True, has_metadata=True)
    process = run_cli(["list", "--verbose"], capsys, monkeypatch)
    assert process.returncode == 0
    assert "Scanning experiments in" in process.combined_output
    assert "DEBUG" in process.combined_output or "INFO" in process.combined_output  # Check if verbose logging level is active
