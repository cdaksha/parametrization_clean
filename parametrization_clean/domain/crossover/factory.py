#!/usr/bin/env python

"""Factory for crossover algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from domain.crossover.single_point import SinglePointCross
from domain.crossover.two_point import TwoPointCross
from domain.crossover.uniform import UniformCross
from domain.crossover.double_pareto import DoubleParetoCross


def crossover_factory(algorithm_name: str):
    """Factory to select crossover type."""
    crossover_types = {
        'single_point': SinglePointCross,
        'two_point': TwoPointCross,
        'uniform': UniformCross,
        'double_pareto': DoubleParetoCross
    }
    return crossover_types[algorithm_name]
