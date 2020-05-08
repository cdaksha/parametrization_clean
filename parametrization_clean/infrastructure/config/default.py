#!/usr/bin/env python

"""Default parameters used for genetic algorithm/neural network."""

# Standard library

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


class DefaultStrategySettings(IStrategySettings):
    selection_strategy = TournamentSelect
    mutation_strategy = GaussianMutate
    crossover_strategy = DoubleParetoCross
    adaptation_strategy = XiaoAdapt
    error_strategy = ReaxError
    initialization_strategy = NakataMutate


class DefaultGeneticAlgorithmSettings(IGeneticAlgorithmSettings):
    population_size = 30
    mutation_rate = 0.2
    crossover_rate = 0.8
    use_elitism = True
    use_adaptation = False
    use_neural_network = False


class DefaultMutationSettings(IMutationSettings):
    gauss_std = [0.01, 0.1]
    gauss_frac = [0.5, 0.5]

    nakata_rand_lower = -1.0
    nakata_rand_higher = 1.0
    nakata_scale = 0.1

    polynomial_eta = 60

    param_bounds = []


class DefaultCrossoverSettings(ICrossoverSettings):
    dpx_alpha = 10
    dpx_beta = 1


class DefaultSelectionSettings(ISelectionSettings):
    tournament_size = 2


class DefaultAdaptationSettings(IAdaptationSettings):
    srinivas_k1 = 1.0
    srinivas_k2 = 0.5
    srinivas_k3 = 1.0
    srinivas_k4 = 0.5
    srinivas_default_mutation_rate = 0.01

    xiao_min_mutation_rate = DefaultGeneticAlgorithmSettings().mutation_rate * 0.8
    xiao_min_crossover_rate = DefaultGeneticAlgorithmSettings().crossover_rate * 0.8
    xiao_scale = 40


class DefaultNeuralNetSettings(INeuralNetSettings):
    verbosity = 2
    train_fraction = 0.8
    num_epochs = 10000
    num_populations_to_train_on = 1
    num_nested_ga_iterations = 1
    minimum_validation_r_squared = 0.95


class DefaultSettings(IAllSettings):
    strategy_settings = DefaultStrategySettings
    ga_settings = DefaultGeneticAlgorithmSettings
    mutation_settings = DefaultMutationSettings
    crossover_settings = DefaultCrossoverSettings
    selection_settings = DefaultSelectionSettings
    adaptation_settings = DefaultAdaptationSettings
    neural_net_settings = DefaultNeuralNetSettings
