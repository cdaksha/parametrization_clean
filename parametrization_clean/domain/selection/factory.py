#!/usr/bin/env python

"""Factory for selection algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.selection.strategy import ISelectionStrategy


class SelectionFactory:
    """Factory class for creating selection algorithm executor."""

    # Internal registry for available crossover methods
    registry = {}

    @classmethod
    def register(cls, algorithm_name: str):
        def inner_wrapper(wrapped_class: ISelectionStrategy):
            # Register algorithm only if it doesn't already exist in the registry
            if algorithm_name not in cls.registry:
                cls.registry[algorithm_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def create_executor(cls, algorithm_name: str) -> ISelectionStrategy:
        return cls.registry[algorithm_name]
