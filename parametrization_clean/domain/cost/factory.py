#!/usr/bin/env python

"""Factory for error calculation algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.cost.strategy import IErrorStrategy
from parametrization_clean.domain.cost.reax_error import ReaxError


class ErrorFactory:
    """Factory class for creating error calculator algorithm executor - RegistryHolder design pattern.
    Classes that implement IErrorStrategy can be registered and utilized through this factory's registry.
    """

    # Internal registry for available crossover methods
    REGISTRY = {}

    @classmethod
    def register(cls, algorithm_name: str, mutation_class):
        """Register a class with a string key."""
        cls.REGISTRY[algorithm_name] = mutation_class
        return mutation_class

    @classmethod
    def create_executor(cls, algorithm_name: str) -> IErrorStrategy:
        return cls.REGISTRY[algorithm_name]


ErrorFactory.register('reax_error', ReaxError)
