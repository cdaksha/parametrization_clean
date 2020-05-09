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

    REGISTRY = {}
    """Internal registry for available adaptation methods. Users can specify from one of the `algorithm_name` strings
    available in the dictionary, mapping `algorithm_name` to the corresponding class implementing that algorithm.
    For example, "xiao" maps to Xiao's adaptation algorithm; users can specify the `adaptation_strategy` in the
    user config.json file to use this algorithm.
    """

    @classmethod
    def register(cls, algorithm_name: str, adaptation_class):
        """Register an adaptation strategy with a string key. Useful for abstraction and dynamic retrieval
        of different algorithms in configuration file. Using this factory, one can easily implement an adaptation
        algorithm (ex. MyAdaptationClass) that follows IAdaptationStrategy, then use
        "AdaptationFactory.register('my_adaptation_class')"
        to generate a corresponding string reference for that adaptation strategy.

        Parameters
        ----------
        algorithm_name: str
            Name that one wishes to assign to the designated `adaptation_class`/algorithm.
        adaptation_class
            Class that one wishes to associate/register with `algorithm_name`.
        Returns
        -------
        adaptation_class
            Same as the `adaptation_class` input parameter.
        """
        cls.REGISTRY[algorithm_name] = adaptation_class
        return adaptation_class

    @classmethod
    def create_executor(cls, algorithm_name: str) -> IAdaptationStrategy:
        return cls.REGISTRY[algorithm_name]


AdaptationFactory.register('srinivas', SrinivasAdapt)
AdaptationFactory.register('xiao', XiaoAdapt)
