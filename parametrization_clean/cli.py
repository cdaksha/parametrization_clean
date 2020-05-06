#!/usr/bin/env python

"""Command-line interface for genetic algorithm + neural network generational propagation."""

# Standard library
import sys
import click

# 3rd party packages

# Local source
from app import run_application


@click.command('cli', short_help="Run one generation of GA.")
@click.option('-g', '--generation_number', default=1, show_default=True, type=click.IntRange(min=1),
              help="Generation number for the genetic algorithm. Default = 1 = first generation.")
@click.option('-t', '--training_path', type=click.Path(), help="File path with reference/training set files.")
@click.option('-p', '--population_path', type=click.Path(), help="File path for generational genetic algorithm output.")
def main(generation_number, training_path, population_path):
    """Command-line interface for genetic algorithm + neural network generational propagation.
    Runs one generation of the GA.
    """
    click.echo("Generation Number: {}".format(generation_number))
    click.echo("Retrieving reference data from: {}".format(training_path))
    click.echo("Outputting generational genetic algorithm data to: {}".format(population_path))
    try:
        response = run_application(generation_number, training_path, population_path)
    except TypeError:
        arguments = [generation_number, training_path, population_path]
        num_expected_arguments = len(arguments)
        num_valid_arguments = sum(1 for arg in arguments if arg)
        raise click.UsageError("Invalid number of arguments passed. Expected {} but received {}. "
                               .format(num_expected_arguments, num_valid_arguments)
                               + 'Try "cli.py --help" for help.')

    click.echo("{}: {}".format(response.type, response.message))
    return response.status_code


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
