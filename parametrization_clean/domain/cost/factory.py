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

    REGISTRY = {}
    """Internal registry for available error calculation methods. Users can specify from one of the
    `algorithm_name` strings available in the dictionary, mapping `algorithm_name` to the corresponding class
    implementing that algorithm.
    For example, "reax_error" maps to the ReaxFF error calculation algorithm;
    users can specify the `error_strategy` in the user config.json file to use this algorithm.
    """

    @classmethod
    def register(cls, algorithm_name: str, error_calculator_class):
        """Register an error calculation strategy with a string key. Useful for abstraction and dynamic retrieval
        of different algorithms in configuration file. Using this factory, one can easily implement an error
        calculation algorithm (ex. MyErrorCalculatorClass) that follows IErrorStrategy, then use
        "ErrorFactory.register('my_error_calculator_class')"
        to generate a corresponding string reference for that error calculation strategy.

        Parameters
        ----------
        algorithm_name: str
            Name that one wishes to assign to the designated `error_calculator_class`/algorithm.
        error_calculator_class
            Class that one wishes to associate/register with `algorithm_name`.
        Returns
        -------
        error_calculator_class
            Same as the `error_calculator_class` input parameter.
        """
        cls.REGISTRY[algorithm_name] = error_calculator_class
        return error_calculator_class

    @classmethod
    def create_executor(cls, algorithm_name: str) -> IErrorStrategy:
        return cls.REGISTRY[algorithm_name]


ErrorFactory.register('reax_error', ReaxError)
