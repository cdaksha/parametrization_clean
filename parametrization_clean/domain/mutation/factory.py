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

    REGISTRY = {}
    """Internal registry for available mutation methods. Users can specify from one of the
    `algorithm_name` strings available in the dictionary, mapping `algorithm_name` to the corresponding class
    implementing that algorithm.
    For example, "gauss" maps to the multi-factor Gaussian algorithm;
    users can specify the `mutation_strategy` in the user config.json file to use this algorithm.
    """

    @classmethod
    def register(cls, algorithm_name: str, mutation_class):
        """Register a mutation strategy with a string key. Useful for abstraction and dynamic retrieval
        of different algorithms in configuration file. Using this factory, one can easily implement a crossover
        algorithm (ex. MyMutationClass) that follows IMutationStrategy, then use
        "MutationFactory.register('my_mutation_class')"
        to generate a corresponding string reference for that mutation strategy.

        Parameters
        ----------
        algorithm_name: str
            Name that one wishes to assign to the designated `mutation_class`/algorithm.
        mutation_class
            Class that one wishes to associate/register with `algorithm_name`.
        Returns
        -------
        mutation_class
            Same as the `mutation_class` input parameter.
        """
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
