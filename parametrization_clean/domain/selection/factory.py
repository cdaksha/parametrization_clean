#!/usr/bin/env python

"""Factory for selection algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.selection.strategy import ISelectionStrategy
from parametrization_clean.domain.selection.tournament import TournamentSelect


class SelectionFactory:
    """Factory class for creating selection algorithm executor - RegistryHolder design pattern.
    Classes that implement ISelectionStrategy can be registered and utilized through this factory's registry.
    """

    # Internal registry for available crossover methods
    REGISTRY = {}

    @classmethod
    def register(cls, algorithm_name: str, mutation_class):
        """Register a class with a string key."""
        cls.REGISTRY[algorithm_name] = mutation_class
        return mutation_class

    @classmethod
    def create_executor(cls, algorithm_name: str) -> ISelectionStrategy:
        return cls.REGISTRY[algorithm_name]


SelectionFactory.register('tournament', TournamentSelect)
