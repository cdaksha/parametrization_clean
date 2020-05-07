#!/usr/bin/env python

"""Factory for adaptation algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from domain.adaptation.strategy import IAdaptationStrategy


class AdaptationFactory:
    """Factory class for creating adaptation algorithm executor."""

    # Internal registry for available crossover methods
    registry = {}

    @classmethod
    def register(cls, algorithm_name: str):
        def inner_wrapper(wrapped_class: IAdaptationStrategy):
            # Register algorithm only if it doesn't already exist in the registry
            if algorithm_name not in cls.registry:
                cls.registry[algorithm_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def create_executor(cls, algorithm_name: str) -> IAdaptationStrategy:
        return cls.registry[algorithm_name]
