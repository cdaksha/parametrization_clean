#!/usr/bin/env python

"""Factory for mutation algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.mutation.strategy import IMutationStrategy
from parametrization_clean.domain.mutation.central_uniform import CentralUniformMutate
from parametrization_clean.domain.mutation.nakata import NakataMutate
from parametrization_clean.domain.mutation.gauss import GaussianMutate
from parametrization_clean.domain.mutation.polynomial import PolynomialMutate


class MutationFactory:
    """Factory class for creating mutation algorithm executor - RegistryHolder design pattern.
    Classes that implement IMutationStrategy can be registered and utilized through this factory's registry.
    """

    # Internal registry for available crossover methods
    REGISTRY = {}

    @classmethod
    def register(cls, algorithm_name: str, mutation_class):
        """Register a class with a string key."""
        cls.REGISTRY[algorithm_name] = mutation_class
        return mutation_class

    @classmethod
    def create_executor(cls, algorithm_name: str) -> IMutationStrategy:
        return cls.REGISTRY[algorithm_name]


# Register all existing mutation algorithms
MutationFactory.register('central_uniform', CentralUniformMutate)
MutationFactory.register('nakata', NakataMutate)
MutationFactory.register('gauss', GaussianMutate)
MutationFactory.register('polynomial', PolynomialMutate)
