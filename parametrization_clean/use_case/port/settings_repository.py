#!/usr/bin/env python

"""
Module that contains interface for repository used to get configuration settings.

__author__ = "Chad Daksha"
"""

# Standard library
import abc

# 3rd party packages

# Local source
from parametrization_clean.domain.selection.strategy import ISelectionStrategy
from parametrization_clean.domain.crossover.strategy import ICrossoverStrategy
from parametrization_clean.domain.mutation.strategy import IMutationStrategy


class IStrategySettings(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def selection_strategy(self) -> ISelectionStrategy:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mutation_strategy(self) -> IMutationStrategy:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def crossover_strategy(self) -> ICrossoverStrategy:
        raise NotImplementedError


class IGeneticAlgorithmSettings(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def population_size(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mutation_rate(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def crossover_rate(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def elitism(self) -> bool:
        raise NotImplementedError


class IMutationSettings(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def gauss_std(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nakata_rand_lower(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nakata_rand_higher(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nakata_scale(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def polynomial_eta(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def param_bounds(self):
        raise NotImplementedError


class ICrossoverSettings(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def dpx_alpha(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dpx_beta(self) -> float:
        raise NotImplementedError


class ISelectionSettings(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def tournament_size(self) -> int:
        raise NotImplementedError


class IAllSettings(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def strategy_settings(self) -> IStrategySettings:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ga_settings(self) -> IGeneticAlgorithmSettings:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mutation_settings(self) -> IMutationSettings:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def crossover_settings(self) -> ICrossoverSettings:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def selection_settings(self) -> ISelectionSettings:
        raise NotImplementedError
