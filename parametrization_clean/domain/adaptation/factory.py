#!/usr/bin/env python

"""Factory for adaptation algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.adaptation.strategy import IAdaptationStrategy
from parametrization_clean.domain.adaptation.srinivas import SrinivasAdapt
from parametrization_clean.domain.adaptation.xiao import XiaoAdapt


class AdaptationFactory:
    """Factory class for creating adaptation algorithm executor - RegistryHolder design pattern.
    Classes that implement IAdaptationStrategy can be registered and utilized through this factory's registry.
    """

    # Internal registry for available crossover methods
    REGISTRY = {}

    @classmethod
    def register(cls, algorithm_name: str, mutation_class):
        """Register a class with a string key."""
        cls.REGISTRY[algorithm_name] = mutation_class
        return mutation_class

    @classmethod
    def create_executor(cls, algorithm_name: str) -> IAdaptationStrategy:
        return cls.REGISTRY[algorithm_name]


AdaptationFactory.register('srinivas', SrinivasAdapt)
AdaptationFactory.register('xiao', XiaoAdapt)
