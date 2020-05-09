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

    REGISTRY = {}
    """Internal registry for available selection methods. Users can specify from one of the
    `algorithm_name` strings available in the dictionary, mapping `algorithm_name` to the corresponding class
    implementing that algorithm.
    For example, "tournament" maps to the Tournament selection algorithm;
    users can specify the `selection_strategy` in the user config.json file to use this algorithm.

    CURRENTLY, ONLY THE TOURNAMENT SELECTION ALGORITHM HAS BEEN IMPLEMENTED AND IS SUPPORTED.
    """

    @classmethod
    def register(cls, algorithm_name: str, selection_class):
        """Register a selection strategy with a string key. Useful for abstraction and dynamic retrieval
        of different algorithms in configuration file. Using this factory, one can easily implement a crossover
        algorithm (ex. MySelectionClass) that follows ISelectionStrategy, then use
        "SelectionFactory.register('my_selection_class_name')"
        to generate a corresponding string reference for that selection strategy.

        Parameters
        ----------
        algorithm_name: str
            Name that one wishes to assign to the designated `selection_class`/algorithm.
        selection_class
            Class that one wishes to associate/register with `algorithm_name`.
        Returns
        -------
        mutation_class
            Same as the `selection_class` input parameter.
        """
        cls.REGISTRY[algorithm_name] = selection_class
        return selection_class

    @classmethod
    def create_executor(cls, algorithm_name: str) -> ISelectionStrategy:
        return cls.REGISTRY[algorithm_name]


SelectionFactory.register('tournament', TournamentSelect)
