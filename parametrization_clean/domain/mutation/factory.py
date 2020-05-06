#!/usr/bin/env python

"""Factory for mutation algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.mutation.nakata import NakataMutate
from parametrization_clean.domain.mutation.central_uniform import CentralUniformMutate
from parametrization_clean.domain.mutation.polynomial import PolynomialMutate
from parametrization_clean.domain.mutation.gauss import GaussianMutate


def mutation_factory(algorithm_name: str):
    """Factory to select mutation type."""
    mutation_types = {
        'nakata': NakataMutate,
        'central_uniform': CentralUniformMutate,
        'polynomial': PolynomialMutate,
        'gauss': GaussianMutate,
    }
    return mutation_types[algorithm_name]
