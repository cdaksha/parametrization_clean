#!/usr/bin/env python

"""Tests for `parametrization_clean` package."""

import os
import shutil

import pytest
from click.testing import CliRunner

from parametrization_clean.cli import main


@pytest.mark.usefixtures("training_set_dir_path", "cli_output_path", "reax_output_dir_path")
def test_command_line_interface(training_set_dir_path, cli_output_path, reax_output_dir_path):
    """Test the CLI."""
    training_path = str(training_set_dir_path)
    population_path = str(cli_output_path)
    config_path = str(os.path.join(os.path.abspath(os.path.join(__file__, "../../")),
                                   "tests", "integration", "config", "cli_config.json"))
    generation_number = 1

    runner = CliRunner()
    result = runner.invoke(main, '--generation_number {} --training_path "{}" --population_path "{}" --config_path "{}"'
                           .format(generation_number, training_path, population_path, config_path))
    print(result.output)
    assert result.exit_code == 0
    assert "Generation Number: 1\n" in result.output
    assert "Retrieving reference data from: {}".format(training_path) in result.output
    assert "Outputting generational genetic algorithm data to: {}".format(population_path) in result.output
    assert "Retrieving user configuration from: {}".format(config_path) in result.output
    assert "{}: {}".format("SUCCESS", "Generation successfully written at {}".format(population_path)) in result.output
    assert not result.exception

    help_result = runner.invoke(main, ['--help'])
    assert help_result.exit_code == 0
    assert "Usage: cli [OPTIONS]" in help_result.output
    assert "Command-line interface for genetic algorithm + neural network generational\n  propagation" in help_result.output
    assert "-g, --generation_number" in help_result.output
    assert "-t, --training_path" in help_result.output
    assert "-p, --population_path" in help_result.output
    assert not help_result.exception

    # Teardown
    shutil.rmtree(os.path.join(population_path, "generation-1"))

    population_path = str(reax_output_dir_path)
    result = runner.invoke(main, '--generation_number {} --training_path "{}" --population_path "{}" --config_path "{}"'
                           .format(3, training_path, population_path, config_path))
    assert result.exit_code == 0
    assert "Generation Number: 3\n" in result.output
    assert "Retrieving reference data from: {}".format(training_path) in result.output
    assert "Outputting generational genetic algorithm data to: {}".format(population_path) in result.output
    assert "Retrieving user configuration from: {}".format(config_path) in result.output
    assert "{}: {}".format("SUCCESS", "Generation successfully written at {}".format(population_path)) in result.output
    assert not result.exception

    # Teardown
    shutil.rmtree(os.path.join(population_path, "generation-3"))

    result = runner.invoke(main, '')
    print(result)
    assert result.exception
