#!/usr/bin/env python

"""Default parameters used for genetic algorithm/neural network."""

# Standard library
from typing import List
# 3rd party packages

# Local source
from parametrization_clean.domain.selection.tournament import TournamentSelect
from parametrization_clean.domain.mutation.gauss import GaussianMutate
from parametrization_clean.domain.crossover.double_pareto import DoubleParetoCross
from parametrization_clean.domain.adaptation.xiao import XiaoAdapt
from parametrization_clean.domain.cost.reax_error import ReaxError
from parametrization_clean.domain.mutation.nakata import NakataMutate
from parametrization_clean.use_case.port.settings_repository import (IStrategySettings,
                                                                     IGeneticAlgorithmSettings,
                                                                     IMutationSettings,
                                                                     ICrossoverSettings,
                                                                     ISelectionSettings,
                                                                     IAdaptationSettings,
                                                                     INeuralNetSettings,
                                                                     IAllSettings)


class DefaultSettings(IAllSettings):

    def __init__(self):
        self._strategy_settings = DefaultStrategySettings()
        self._ga_settings = DefaultGeneticAlgorithmSettings()
        self._mutation_settings = DefaultMutationSettings()
        self._crossover_settings = DefaultCrossoverSettings()
        self._selection_settings = DefaultSelectionSettings()
        self._adaptation_settings = DefaultAdaptationSettings()
        self._neural_net_settings = DefaultNeuralNetSettings()

    @property
    def strategy_settings(self) -> IStrategySettings:
        return self._strategy_settings

    @property
    def ga_settings(self) -> IGeneticAlgorithmSettings:
        return self._ga_settings

    @property
    def mutation_settings(self) -> IMutationSettings:
        return self._mutation_settings

    @property
    def crossover_settings(self) -> ICrossoverSettings:
        return self._crossover_settings

    @property
    def selection_settings(self) -> ISelectionSettings:
        return self._selection_settings

    @property
    def adaptation_settings(self) -> IAdaptationSettings:
        return self._adaptation_settings

    @property
    def neural_net_settings(self) -> INeuralNetSettings:
        return self._neural_net_settings


class DefaultStrategySettings(IStrategySettings):

    def __init__(self):
        self._selection_strategy = TournamentSelect
        self._mutation_strategy = GaussianMutate
        self._crossover_strategy = DoubleParetoCross
        self._adaptation_strategy = XiaoAdapt
        self._error_strategy = ReaxError
        self._initialization_strategy = NakataMutate

    @property
    def selection_strategy(self):
        return self._selection_strategy

    @property
    def mutation_strategy(self):
        return self._mutation_strategy

    @property
    def crossover_strategy(self):
        return self._crossover_strategy

    @property
    def adaptation_strategy(self):
        return self._adaptation_strategy

    @property
    def error_strategy(self):
        return self._error_strategy

    @property
    def initialization_strategy(self):
        return self._initialization_strategy


class DefaultGeneticAlgorithmSettings(IGeneticAlgorithmSettings):

    def __init__(self):
        self._population_size = 30
        self._mutation_rate = 0.2
        self._crossover_rate = 0.8
        self._use_elitism = True
        self._use_adaptation = False
        self._use_neural_network = False

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate

    @property
    def crossover_rate(self) -> float:
        return self._crossover_rate

    @property
    def use_elitism(self) -> bool:
        return self._use_elitism

    @property
    def use_adaptation(self) -> bool:
        return self._use_adaptation

    @property
    def use_neural_network(self) -> bool:
        return self._use_neural_network


class DefaultMutationSettings(IMutationSettings):

    def __init__(self):
        self._gauss_std = [0.01, 0.1]
        self._gauss_frac = [0.5, 0.5]
        self._nakata_rand_lower = -1.0
        self._nakata_rand_higher = 1.0
        self._nakata_scale = 0.1
        self._polynomial_eta = 60
        self._param_bounds = []

    @property
    def gauss_std(self) -> List[float]:
        return self._gauss_std

    @property
    def gauss_frac(self) -> List[float]:
        return self._gauss_frac

    @property
    def nakata_rand_lower(self) -> float:
        return self._nakata_rand_lower

    @property
    def nakata_rand_higher(self) -> float:
        return self._nakata_rand_higher

    @property
    def nakata_scale(self) -> float:
        return self._nakata_scale

    @property
    def polynomial_eta(self) -> float:
        return self._polynomial_eta

    @property
    def param_bounds(self):
        return self._param_bounds


class DefaultCrossoverSettings(ICrossoverSettings):

    def __init__(self):
        self._dpx_alpha = 10
        self._dpx_beta = 1

    @property
    def dpx_alpha(self) -> float:
        return self._dpx_alpha

    @property
    def dpx_beta(self) -> float:
        return self._dpx_beta


class DefaultSelectionSettings(ISelectionSettings):

    def __init__(self):
        self._tournament_size = 2

    @property
    def tournament_size(self) -> int:
        return self._tournament_size


class DefaultAdaptationSettings(IAdaptationSettings):

    def __init__(self):
        self._srinivas_k1 = 1.0
        self._srinivas_k2 = 0.5
        self._srinivas_k3 = 1.0
        self._srinivas_k4 = 0.5
        self._srinivas_default_mutation_rate = 0.01
        self._xiao_min_crossover_rate = DefaultGeneticAlgorithmSettings().crossover_rate * 0.8
        self._xiao_min_mutation_rate = DefaultGeneticAlgorithmSettings().mutation_rate * 0.8
        self._xiao_scale = 40

    @property
    def srinivas_k1(self) -> float:
        return self._srinivas_k1

    @property
    def srinivas_k2(self) -> float:
        return self._srinivas_k2

    @property
    def srinivas_k3(self) -> float:
        return self._srinivas_k3

    @property
    def srinivas_k4(self) -> float:
        return self._srinivas_k4

    @property
    def srinivas_default_mutation_rate(self) -> float:
        return self._srinivas_default_mutation_rate

    @property
    def xiao_min_crossover_rate(self) -> float:
        return self._xiao_min_crossover_rate

    @property
    def xiao_min_mutation_rate(self) -> float:
        return self._xiao_min_mutation_rate

    @property
    def xiao_scale(self) -> float:
        return self._xiao_scale


class DefaultNeuralNetSettings(INeuralNetSettings):

    def __init__(self):
        self._verbosity = 2
        self._train_fraction = 0.8
        self._num_epochs = 10000
        self._num_populations_to_train_on = 1
        self._num_nested_ga_iterations = 1
        self._minimum_validation_r_squared = 0.95

    @property
    def verbosity(self) -> int:
        return self._verbosity

    @property
    def train_fraction(self) -> float:
        return self._train_fraction

    @property
    def num_epochs(self) -> int:
        return self._num_epochs

    @property
    def num_populations_to_train_on(self) -> int:
        return self._num_populations_to_train_on

    @property
    def num_nested_ga_iterations(self) -> int:
        return self._num_nested_ga_iterations

    @property
    def minimum_validation_r_squared(self) -> float:
        return self._minimum_validation_r_squared
