#!/usr/bin/env python

"""
Module that contains interface for repository used to get configuration settings.

__author__ = "Chad Daksha"
"""

# Standard library
import abc
from typing import List

# 3rd party packages

# Local source
from parametrization_clean.domain.selection.strategy import ISelectionStrategy
from parametrization_clean.domain.crossover.strategy import ICrossoverStrategy
from parametrization_clean.domain.mutation.strategy import IMutationStrategy
from parametrization_clean.domain.adaptation.strategy import IAdaptationStrategy
from parametrization_clean.domain.cost.strategy import IErrorStrategy


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

    @property
    @abc.abstractmethod
    def adaptation_strategy(self) -> IAdaptationStrategy:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def error_strategy(self) -> IErrorStrategy:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def initialization_strategy(self) -> IMutationStrategy:
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
    def use_elitism(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def use_adaptation(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def use_neural_network(self) -> bool:
        raise NotImplementedError


class IMutationSettings(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def gauss_std(self) -> List[float]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gauss_frac(self) -> List[float]:
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


class IAdaptationSettings(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def srinivas_k1(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def srinivas_k2(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def srinivas_k3(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def srinivas_k4(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def srinivas_default_mutation_rate(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def xiao_min_crossover_rate(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def xiao_min_mutation_rate(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def xiao_scale(self) -> float:
        raise NotImplementedError


class INeuralNetSettings(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def verbosity(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def train_fraction(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_epochs(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_populations_to_train_on(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_nested_ga_iterations(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def minimum_validation_r_squared(self) -> float:
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

    @property
    @abc.abstractmethod
    def adaptation_settings(self) -> IAdaptationSettings:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def neural_net_settings(self) -> INeuralNetSettings:
        raise NotImplementedError
