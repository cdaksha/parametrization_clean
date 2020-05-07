#!/usr/bin/env python

"""Command-line interface for genetic algorithm + neural network generational propagation."""

# Standard library
import sys
import click

# 3rd party packages

# Local source
from app import run_application


@click.command('cli', short_help="Run one generation of GA.")
@click.option('-g', '--generation_number', default=1, show_default=True, type=click.IntRange(min=1), required=True,
              help="Generation number for the genetic algorithm. Default = 1 = first generation.")
@click.option('-t', '--training_path', type=click.Path(), required=True,
              help="File path with reference/training set files.")
@click.option('-p', '--population_path', type=click.Path(), required=True,
              help="File path for generational genetic algorithm output.")
@click.option('-c', '--config_path', type=click.Path(),
              help="File path with location of (optional) user configuration file.")
def main(generation_number, training_path, population_path, config_path):
    """Command-line interface for genetic algorithm + neural network generational propagation.
    Runs one generation of the GA.
    """
    # TODO: Make population size a command line option instead of a user configuration file option?
    click.echo("Generation Number: {}".format(generation_number))
    click.echo("Retrieving reference data from: {}".format(training_path))
    click.echo("Retrieving user configuration from: {}".format(config_path if config_path
                                                               else "no user config provided...using DEFAULT config."))
    click.echo("Outputting generational genetic algorithm data to: {}".format(population_path))

    response = run_application(generation_number, training_path, population_path, config_path)

    click.echo("{}: {}".format(response.type, response.message))
    return response.status_code


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
