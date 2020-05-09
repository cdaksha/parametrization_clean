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

    REGISTRY = {}
    """Internal registry for available crossover methods. Users can specify from one of the
    `algorithm_name` strings available in the dictionary, mapping `algorithm_name` to the corresponding class
    implementing that algorithm.
    For example, "single_point" maps to the single point crossover algorithm (for real-valued GAs);
    users can specify the `crossover_strategy` in the user config.json file to use this algorithm.
    """

    @classmethod
    def register(cls, algorithm_name: str, crossover_class):
        """Register a crossover strategy with a string key. Useful for abstraction and dynamic retrieval
        of different algorithms in configuration file. Using this factory, one can easily implement a crossover
        algorithm (ex. MyCrossoverClass) that follows ICrossoverStrategy, then use
        "CrossoverFactory.register('my_crossover_class_name')"
        to generate a corresponding string reference for that crossover strategy.

        Parameters
        ----------
        algorithm_name: str
            Name that one wishes to assign to the designated `crossover_class`/algorithm.
        crossover_class
            Class that one wishes to associate/register with `algorithm_name`.
        Returns
        -------
        crossover_class
            Same as the `crossover_class` input parameter.
        """
        cls.REGISTRY[algorithm_name] = crossover_class
        return crossover_class

    @classmethod
    def create_executor(cls, algorithm_name: str) -> ICrossoverStrategy:
        return cls.REGISTRY[algorithm_name]


CrossoverFactory.register('single_point', SinglePointCross)
CrossoverFactory.register('two_point', TwoPointCross)
CrossoverFactory.register('uniform', UniformCross)
CrossoverFactory.register('double_pareto', DoubleParetoCross)
