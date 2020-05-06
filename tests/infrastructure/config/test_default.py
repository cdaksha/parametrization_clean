
# Standard library

# 3rd party packages

# Local source
from domain.selection.tournament import TournamentSelect
from domain.mutation.gauss import GaussianMutate
from domain.crossover.double_pareto import DoubleParetoCross
from domain.adaptation.xiao import XiaoAdapt
from domain.cost.reax_error import ReaxError
from domain.mutation.nakata import NakataMutate
from infrastructure.config.default import DefaultSettings


def test_all_settings_init():
    default_settings = DefaultSettings()

    assert default_settings.strategy_settings.selection_strategy == TournamentSelect
    assert default_settings.strategy_settings.error_strategy == ReaxError
    assert default_settings.strategy_settings.adaptation_strategy == XiaoAdapt
    assert default_settings.strategy_settings.crossover_strategy == DoubleParetoCross
    assert default_settings.strategy_settings.mutation_strategy == GaussianMutate
    assert default_settings.strategy_settings.initialization_strategy == NakataMutate

    assert default_settings.ga_settings.population_size == 30
    assert default_settings.ga_settings.mutation_rate == 0.2
    assert default_settings.ga_settings.crossover_rate == 0.8
    assert default_settings.ga_settings.use_elitism
    assert not default_settings.ga_settings.use_adaptation

    assert default_settings.mutation_settings.gauss_std == [0.01, 0.1]
    assert default_settings.mutation_settings.gauss_frac == [0.5, 0.5]
    assert default_settings.mutation_settings.nakata_rand_lower == -1.0
    assert default_settings.mutation_settings.nakata_rand_higher == 1.0
    assert default_settings.mutation_settings.nakata_scale == 0.1
    assert default_settings.mutation_settings.polynomial_eta == 60
    assert not default_settings.mutation_settings.param_bounds

    assert default_settings.crossover_settings.dpx_alpha == 10
    assert default_settings.crossover_settings.dpx_beta == 1

    assert default_settings.selection_settings.tournament_size == 2

    assert default_settings.adaptation_settings.srinivas_k1 == 1.0
    assert default_settings.adaptation_settings.srinivas_k2 == 0.5
    assert default_settings.adaptation_settings.srinivas_k3 == 1.0
    assert default_settings.adaptation_settings.srinivas_k4 == 0.5
    assert default_settings.adaptation_settings.srinivas_default_mutation_rate == 0.01
    assert default_settings.adaptation_settings.xiao_min_crossover_rate == 0.8 * 0.8
    assert default_settings.adaptation_settings.xiao_min_mutation_rate == 0.2 * 0.8
    assert default_settings.adaptation_settings.xiao_scale == 40

    assert default_settings.neural_net_settings.verbosity == 2
    assert default_settings.neural_net_settings.train_fraction == 0.8
    assert default_settings.neural_net_settings.num_epochs == 10000
    assert default_settings.neural_net_settings.num_populations_to_train_on == 1
    assert default_settings.neural_net_settings.num_nested_ga_iterations == 1
