import pytest
import yaml

from pasim.cli import _check_params_against_template


def test_check_params_against_template_missing(tmp_path, monkeypatch, caplog):
    """Verify ValueError is raised when parameters are missing."""
    # Create a dummy template
    template_path = tmp_path / "params_template.yaml"
    template_content = {"n_runs": 1, "total_ticks": 100, "new_param": "foo"}
    template_path.write_text(yaml.dump(template_content))

    # Create a user params file missing 'new_param'
    user_params_path = tmp_path / "params.yaml"
    user_content = {"n_runs": 1, "total_ticks": 100}
    user_params_path.write_text(yaml.dump(user_content))

    # Mock the global PARAMS_TEMPLATE_PATH
    import pasim.cli

    monkeypatch.setattr(pasim.cli, "PARAMS_TEMPLATE_PATH", template_path)

    with pytest.raises(ValueError) as excinfo:
        _check_params_against_template(user_params_path)

    assert "missing in your file: ['new_param']" in str(excinfo.value)


def test_check_params_against_template_extra(tmp_path, monkeypatch, caplog):
    """Verify warnings are issued when extra parameters are present."""
    template_path = tmp_path / "params_template.yaml"
    template_content = {"n_runs": 1}
    template_path.write_text(yaml.dump(template_content))

    user_params_path = tmp_path / "params.yaml"
    user_content = {"n_runs": 1, "extra_param": "bar"}
    user_params_path.write_text(yaml.dump(user_content))

    import pasim.cli

    monkeypatch.setattr(pasim.cli, "PARAMS_TEMPLATE_PATH", template_path)

    with caplog.at_level("WARNING"):
        _check_params_against_template(user_params_path)

    assert "contains keys not present in the template: ['extra_param']" in caplog.text


def test_check_params_against_template_match(tmp_path, monkeypatch, caplog):
    """Verify no warnings when parameters match exactly."""
    template_path = tmp_path / "params_template.yaml"
    template_content = {"n_runs": 1}
    template_path.write_text(yaml.dump(template_content))

    user_params_path = tmp_path / "params.yaml"
    user_content = {"n_runs": 1}
    user_params_path.write_text(yaml.dump(user_content))

    import pasim.cli

    monkeypatch.setattr(pasim.cli, "PARAMS_TEMPLATE_PATH", template_path)

    with caplog.at_level("WARNING"):
        _check_params_against_template(user_params_path)

    assert caplog.text == ""
