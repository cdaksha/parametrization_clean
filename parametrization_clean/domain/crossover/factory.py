#!/usr/bin/env python

"""Factory for crossover algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.crossover.single_point import SinglePointCross
from parametrization_clean.domain.crossover.two_point import TwoPointCross
from parametrization_clean.domain.crossover.uniform import UniformCross
from parametrization_clean.domain.crossover.double_pareto import DoubleParetoCross


def crossover_factory(algorithm_name: str):
    """Factory to select crossover type."""
    crossover_types = {
        'single_point': SinglePointCross,
        'two_point': TwoPointCross,
        'uniform': UniformCross,
        'double_pareto': DoubleParetoCross
    }
    return crossover_types[algorithm_name]
