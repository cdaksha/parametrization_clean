#!/usr/bin/env python

"""Factory for crossover algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.crossover.strategy import ICrossoverStrategy
from parametrization_clean.domain.crossover.single_point import SinglePointCross
from parametrization_clean.domain.crossover.two_point import TwoPointCross
from parametrization_clean.domain.crossover.uniform import UniformCross
from parametrization_clean.domain.crossover.double_pareto import DoubleParetoCross


class CrossoverFactory:
    """Factory class for creating crossover algorithm executor - RegistryHolder design pattern.
    Classes that implement ICrossoverStrategy can be registered and utilized through this factory's registry.
    """

    # Internal registry for available crossover methods
    REGISTRY = {}

    @classmethod
    def register(cls, algorithm_name: str, mutation_class):
        """Register a class with a string key."""
        cls.REGISTRY[algorithm_name] = mutation_class
        return mutation_class

    @classmethod
    def create_executor(cls, algorithm_name: str) -> ICrossoverStrategy:
        return cls.REGISTRY[algorithm_name]


CrossoverFactory.register('single_point', SinglePointCross)
CrossoverFactory.register('two_point', TwoPointCross)
CrossoverFactory.register('uniform', UniformCross)
CrossoverFactory.register('double_pareto', DoubleParetoCross)
