
# Standard library
import os

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.selection.tournament import TournamentSelect
from parametrization_clean.domain.mutation.gauss import GaussianMutate
from parametrization_clean.domain.crossover.double_pareto import DoubleParetoCross
from parametrization_clean.domain.crossover.single_point import SinglePointCross
from parametrization_clean.domain.adaptation.xiao import XiaoAdapt
from parametrization_clean.domain.cost.reax_error import ReaxError
from parametrization_clean.domain.mutation.nakata import NakataMutate
from parametrization_clean.domain.mutation.central_uniform import CentralUniformMutate
from parametrization_clean.infrastructure.config.local import UserSettings
from parametrization_clean.infrastructure.exception.exception import ConfigurationError


PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../../../../"))


def test_user_settings_init_with_empty_file():
    config_file_path = os.path.join(PROJECT_ROOT, "tests", "integration",
                                    "config", "empty_config.json")

    with pytest.raises(ConfigurationError):
        UserSettings(config_file_path)


def test_user_settings_init_with_nonexistent_file():
    config_file_path = os.path.join(PROJECT_ROOT, "tests", "integration",
                                    "config", "nonexistent_config.json")
    user_settings = UserSettings(config_file_path)

    assert user_settings.strategy_settings.selection_strategy == TournamentSelect
    assert user_settings.strategy_settings.error_strategy == ReaxError
    assert user_settings.strategy_settings.adaptation_strategy == XiaoAdapt
    assert user_settings.strategy_settings.crossover_strategy == DoubleParetoCross
    assert user_settings.strategy_settings.mutation_strategy == GaussianMutate
    assert user_settings.strategy_settings.initialization_strategy == NakataMutate

    assert user_settings.ga_settings.population_size == 30
    assert user_settings.ga_settings.mutation_rate == 0.2
    assert user_settings.ga_settings.crossover_rate == 0.8
    assert user_settings.ga_settings.use_elitism
    assert not user_settings.ga_settings.use_adaptation

    assert user_settings.mutation_settings.gauss_std == [0.01, 0.1]
    assert user_settings.mutation_settings.gauss_frac == [0.5, 0.5]
    assert user_settings.mutation_settings.nakata_rand_lower == -1.0
    assert user_settings.mutation_settings.nakata_rand_higher == 1.0
    assert user_settings.mutation_settings.nakata_scale == 0.1
    assert user_settings.mutation_settings.polynomial_eta == 60
    assert not user_settings.mutation_settings.param_bounds

    assert user_settings.crossover_settings.dpx_alpha == 10
    assert user_settings.crossover_settings.dpx_beta == 1

    assert user_settings.selection_settings.tournament_size == 2

    assert user_settings.adaptation_settings.srinivas_k1 == 1.0
    assert user_settings.adaptation_settings.srinivas_k2 == 0.5
    assert user_settings.adaptation_settings.srinivas_k3 == 1.0
    assert user_settings.adaptation_settings.srinivas_k4 == 0.5
    assert user_settings.adaptation_settings.srinivas_default_mutation_rate == 0.01
    assert user_settings.adaptation_settings.xiao_min_crossover_rate == 0.8 * 0.8
    assert user_settings.adaptation_settings.xiao_min_mutation_rate == 0.2 * 0.8
    assert user_settings.adaptation_settings.xiao_scale == 40

    assert user_settings.neural_net_settings.verbosity == 2
    assert user_settings.neural_net_settings.train_fraction == 0.8
    assert user_settings.neural_net_settings.num_epochs == 10000
    assert user_settings.neural_net_settings.num_populations_to_train_on == 1
    assert user_settings.neural_net_settings.num_nested_ga_iterations == 1


def test_user_settings_init_with_some_parameters_specified():
    config_file_path = os.path.join(PROJECT_ROOT, "tests", "integration",
                                    "config", "config.json")
    user_settings = UserSettings(config_file_path)

    assert user_settings.strategy_settings.selection_strategy == TournamentSelect
    assert user_settings.strategy_settings.error_strategy == ReaxError
    assert user_settings.strategy_settings.adaptation_strategy == XiaoAdapt
    assert user_settings.strategy_settings.crossover_strategy == SinglePointCross
    assert user_settings.strategy_settings.mutation_strategy == GaussianMutate
    assert user_settings.strategy_settings.initialization_strategy == CentralUniformMutate

    assert user_settings.ga_settings.population_size == 10
    assert user_settings.ga_settings.mutation_rate == 0.2
    assert user_settings.ga_settings.crossover_rate == 0.8
    assert not user_settings.ga_settings.use_elitism
    assert user_settings.ga_settings.use_adaptation

    assert user_settings.mutation_settings.gauss_std == [0.01, 0.1, 1.0]
    assert user_settings.mutation_settings.gauss_frac == [0.25, 0.5, 0.25]
    assert user_settings.mutation_settings.nakata_rand_lower == -1.0
    assert user_settings.mutation_settings.nakata_rand_higher == 1.0
    assert user_settings.mutation_settings.nakata_scale == 0.1
    assert user_settings.mutation_settings.polynomial_eta == 60
    assert not user_settings.mutation_settings.param_bounds

    assert user_settings.crossover_settings.dpx_alpha == 10
    assert user_settings.crossover_settings.dpx_beta == 100

    assert user_settings.selection_settings.tournament_size == 3

    assert user_settings.adaptation_settings.srinivas_k1 == 1.0
    assert user_settings.adaptation_settings.srinivas_k2 == 0.5
    assert user_settings.adaptation_settings.srinivas_k3 == 1.0
    assert user_settings.adaptation_settings.srinivas_k4 == 0.5
    assert user_settings.adaptation_settings.srinivas_default_mutation_rate == 0.05
    assert user_settings.adaptation_settings.xiao_min_crossover_rate == 0.4
    assert user_settings.adaptation_settings.xiao_min_mutation_rate == 0.1
    assert user_settings.adaptation_settings.xiao_scale == 4

    assert user_settings.neural_net_settings.verbosity == 0
    assert user_settings.neural_net_settings.train_fraction == 0.8
    assert user_settings.neural_net_settings.num_epochs == 10000
    assert user_settings.neural_net_settings.num_populations_to_train_on == 10
    assert user_settings.neural_net_settings.num_nested_ga_iterations == 100
