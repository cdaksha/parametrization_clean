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


class IStrategySettings(metaclass=abc.ABC):
    selection_strategy: ISelectionStrategy = NotImplemented
    mutation_strategy: IMutationStrategy = NotImplemented
    crossover_strategy: ICrossoverStrategy = NotImplemented
    adaptation_strategy: IAdaptationStrategy = NotImplemented
    error_strategy: IErrorStrategy = NotImplemented
    initialization_strategy: IMutationStrategy = NotImplemented


class IGeneticAlgorithmSettings(metaclass=abc.ABC):
    population_size: int = NotImplemented
    mutation_rate: float = NotImplemented
    crossover_rate: float = NotImplemented
    use_elitism: bool = NotImplemented
    use_adaptation: bool = NotImplemented
    use_neural_network: bool = NotImplemented


class IMutationSettings(metaclass=abc.ABC):
    gauss_std: List[float] = NotImplemented
    gauss_frac: List[float] = NotImplemented

    nakata_rand_lower: float = NotImplemented
    nakata_rand_higher: float = NotImplemented
    nakata_scale: float = NotImplemented

    polynomial_eta: float = NotImplemented

    param_bounds: List[List[float]] = NotImplemented


class ICrossoverSettings(metaclass=abc.ABC):
    dpx_alpha: float = NotImplemented
    dpx_beta: float = NotImplemented


class ISelectionSettings(metaclass=abc.ABC):
    tournament_size: int = NotImplemented


class IAdaptationSettings(metaclass=abc.ABC):
    srinivas_k1: float = NotImplemented
    srinivas_k2: float = NotImplemented
    srinivas_k3: float = NotImplemented
    srinivas_k4: float = NotImplemented
    srinivas_default_mutation_rate: float = NotImplemented

    xiao_min_crossover_rate: float = NotImplemented
    xiao_min_mutation_rate: float = NotImplemented
    xiao_scale: float = NotImplemented


class INeuralNetSettings(metaclass=abc.ABC):
    verbosity: int = NotImplemented
    train_fraction: float = NotImplemented
    num_epochs: int = NotImplemented
    num_populations_to_train_on: int = NotImplemented
    num_nested_ga_iterations: int = NotImplemented
    minimum_validation_r_squared: float = NotImplemented


class IAllSettings(metaclass=abc.ABC):
    strategy_settings: IStrategySettings = NotImplemented
    ga_settings: IGeneticAlgorithmSettings = NotImplemented
    mutation_settings: IMutationSettings = NotImplemented
    crossover_settings: ICrossoverSettings = NotImplemented
    selection_settings: ISelectionSettings = NotImplemented
    neural_net_settings: INeuralNetSettings = NotImplemented
