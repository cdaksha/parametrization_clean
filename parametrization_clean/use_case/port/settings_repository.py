#!/usr/bin/env python

"""Module that contains interface for settings repository, used to get configuration settings.
All settings are subdivided into different categories based on groupings/relevance.
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


class IStrategySettings(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.selection_strategy: ISelectionStrategy = NotImplemented
        self.mutation_strategy: IMutationStrategy = NotImplemented
        self.crossover_strategy: ICrossoverStrategy = NotImplemented
        self.adaptation_strategy: IAdaptationStrategy = NotImplemented
        self.error_strategy: IErrorStrategy = NotImplemented
        self.initialization_strategy: IMutationStrategy = NotImplemented


class IGeneticAlgorithmSettings:

    @abc.abstractmethod
    def __init__(self):
        self.population_size: int = NotImplemented
        self.mutation_rate: float = NotImplemented
        self.crossover_rate: float = NotImplemented
        self.use_elitism: bool = NotImplemented
        self.use_adaptation: bool = NotImplemented
        self.use_neural_network: bool = NotImplemented


class IMutationSettings(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.gauss_std: List[float] = NotImplemented
        self.gauss_frac: List[float] = NotImplemented

        self.nakata_rand_lower: float = NotImplemented
        self.nakata_rand_higher: float = NotImplemented
        self.nakata_scale: float = NotImplemented

        self.polynomial_eta: float = NotImplemented

        self.param_bounds: List[List[float]] = NotImplemented


class ICrossoverSettings(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.dpx_alpha: float = NotImplemented
        self.dpx_beta: float = NotImplemented


class ISelectionSettings(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.tournament_size: int = NotImplemented


class IAdaptationSettings(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.srinivas_k1: float = NotImplemented
        self.srinivas_k2: float = NotImplemented
        self.srinivas_k3: float = NotImplemented
        self.srinivas_k4: float = NotImplemented
        self.srinivas_default_mutation_rate: float = NotImplemented

        self.xiao_min_crossover_rate: float = NotImplemented
        self.xiao_min_mutation_rate: float = NotImplemented
        self.xiao_scale: float = NotImplemented


class INeuralNetSettings(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.verbosity: int = NotImplemented
        self.train_fraction: float = NotImplemented
        self.num_epochs: int = NotImplemented
        self.num_populations_to_train_on: int = NotImplemented
        self.num_nested_ga_iterations: int = NotImplemented
        self.minimum_validation_r_squared: float = NotImplemented


class IAllSettings(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.strategy_settings: IStrategySettings = NotImplemented
        self.ga_settings: IGeneticAlgorithmSettings = NotImplemented
        self.mutation_settings: IMutationSettings = NotImplemented
        self.crossover_settings: ICrossoverSettings = NotImplemented
        self.selection_settings: ISelectionSettings = NotImplemented
        self.neural_net_settings: INeuralNetSettings = NotImplemented
