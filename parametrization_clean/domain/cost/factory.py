#!/usr/bin/env python

"""Factory for error calculation algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from domain.cost.reax_error import ReaxError


def error_calculator_factory(algorithm_name: str):
    """Factory to select cost/error calculator type."""
    error_calculator_types = {
        'reax_error': ReaxError,
    }
    return error_calculator_types[algorithm_name]
