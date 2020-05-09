#!/usr/bin/env python

"""Contains default config classes with default parameters used for genetic algorithm/neural network, based on the
settings repository port. Note that the attributes implemented map one-to-one with the corresponding JSON key
that would be expected from the user config, if specified. For example, at the highest level, DefaultSettings
is composed of `strategy_settings`, `ga_settings`, ..., `neural_net_settings`. Settings for each of those attributes
would be specified in the JSON as following:

{
    "strategy_settings": {
        ...
    },
    "ga_settings": {
        ...
    },
    ...
    "neural_net_settings": {
        ...
    }
}

For each of those keys in the JSON, then, the corresponding attributes also map directly to the JSON key names expected.
For example, `ga_settings` represents an implementation of IGeneticAlgorithmSettings, which expects several attributes:
`population_size`, `mutation_rate`, `crossover_rate`, and so on. If the user wishes to change, for example, the
population size, their JSON configuration file might look like

{
    "ga_settings": {
        "population_size": 100
    }
}

The user can choose to override as many of the existing attributes as they wish, but reasonable defaults are provided
for the adaptation, selection, crossover, and mutation algorithms, as well as for the neural network. However, IT IS
HIGHLY RECOMMENDED TO SET, AT THE VERY LEAST, THE POPULATION SIZE OF THE GENETIC ALGORITHM BASED ON THE USER'S DESIRES.
"""

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


class DefaultSettings(IAllSettings):

    def __init__(self):
        super().__init__()
        self.strategy_settings = DefaultStrategySettings()
        self.ga_settings = DefaultGeneticAlgorithmSettings()
        self.mutation_settings = DefaultMutationSettings()
        self.crossover_settings = DefaultCrossoverSettings()
        self.selection_settings = DefaultSelectionSettings()
        self.adaptation_settings = DefaultAdaptationSettings()
        self.neural_net_settings = DefaultNeuralNetSettings()


class DefaultStrategySettings(IStrategySettings):

    def __init__(self):
        super().__init__()
        self.selection_strategy = TournamentSelect
        self.mutation_strategy = GaussianMutate
        self.crossover_strategy = DoubleParetoCross
        self.adaptation_strategy = XiaoAdapt
        self.error_strategy = ReaxError
        self.initialization_strategy = NakataMutate


class DefaultGeneticAlgorithmSettings(IGeneticAlgorithmSettings):

    def __init__(self):
        super().__init__()
        self.population_size = 30
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        self.use_elitism = True
        self.use_adaptation = False
        self.use_neural_network = False


class DefaultMutationSettings(IMutationSettings):

    def __init__(self):
        super().__init__()
        self.gauss_std = [0.01, 0.1]
        self.gauss_frac = [0.5, 0.5]

        self.nakata_rand_lower = -1.0
        self.nakata_rand_higher = 1.0
        self.nakata_scale = 0.1

        self.polynomial_eta = 60

        # TODO: Currently causes issues with central uniform mutate -> no mutations caused...
        # Either set equal to None or raise Exception if central uniform mutate called with param bounds that aren't set
        self.param_bounds = []


class DefaultCrossoverSettings(ICrossoverSettings):

    def __init__(self):
        super().__init__()
        self.dpx_alpha = 10
        self.dpx_beta = 1


class DefaultSelectionSettings(ISelectionSettings):

    def __init__(self):
        super().__init__()
        self.tournament_size = 2


class DefaultAdaptationSettings(IAdaptationSettings):

    def __init__(self):
        super().__init__()
        self.srinivas_k1 = 1.0
        self.srinivas_k2 = 0.5
        self.srinivas_k3 = 1.0
        self.srinivas_k4 = 0.5
        self.srinivas_default_mutation_rate = 0.01

        self.xiao_min_mutation_rate = DefaultGeneticAlgorithmSettings().mutation_rate * 0.8
        self.xiao_min_crossover_rate = DefaultGeneticAlgorithmSettings().crossover_rate * 0.8
        self.xiao_scale = 40


class DefaultNeuralNetSettings(INeuralNetSettings):

    def __init__(self):
        super().__init__()
        self.verbosity = 2
        self.train_fraction = 0.8
        self.num_epochs = 10000
        self.num_populations_to_train_on = 1
        self.num_nested_ga_iterations = 1
        self.minimum_validation_r_squared = 0.95
