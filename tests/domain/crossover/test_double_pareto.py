
# Standard library
from unittest.mock import patch

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.crossover.double_pareto import DoubleParetoCross
from parametrization_clean.domain.root_individual import RootIndividual


@pytest.fixture()
@pytest.mark.usefixtures('dft_energies', 'weights', 'root_ffield', 'param_keys')
def get_root_individual(dft_energies, weights, root_ffield, param_keys):
    return RootIndividual(dft_energies, weights, root_ffield, param_keys)


@pytest.fixture()
def double_pareto_params():
    alpha = 10
    beta = 1
    return alpha, beta


def get_true_children_params(parent1_params, parent2_params, modified_beta):
    child1_true_params = [((parent1_param + parent2_param) + modified_beta * abs(parent1_param - parent2_param)) / 2
                          for parent1_param, parent2_param in zip(parent1_params, parent2_params)]
    child2_true_params = [((parent1_param + parent2_param) - modified_beta * abs(parent1_param - parent2_param)) / 2
                          for parent1_param, parent2_param in zip(parent1_params, parent2_params)]
    return child1_true_params, child2_true_params


@patch('random.uniform')
@pytest.mark.usefixtures('get_individuals')
def test_double_pareto_1(uniform_mock, get_root_individual, get_individuals, double_pareto_params):
    alpha, beta = double_pareto_params
    uniform_mock.return_value = 0.75
    modified_beta = alpha * beta * (1 - (2 * 0.75) ** (-1 / alpha))
    parent1 = get_individuals[0]
    parent2 = get_individuals[1]
    child1_true_params, child2_true_params = get_true_children_params(parent1.params, parent2.params, modified_beta)
    child1, child2 = DoubleParetoCross.crossover(parent1, parent2, root_individual=get_root_individual,
                                                 dpx_alpha=alpha, dpx_beta=beta)
    assert child1.params == child1_true_params
    assert child2.params == child2_true_params


@patch('random.uniform')
@pytest.mark.usefixtures('get_individuals')
def test_double_pareto_2(uniform_mock, get_root_individual, get_individuals, double_pareto_params):
    alpha, beta = double_pareto_params
    uniform_mock.return_value = 0.25
    modified_beta = alpha * beta * ((1 - (2 * 0.25)) ** (-1 / alpha) - 1)
    parent1, parent2 = get_individuals[0], get_individuals[1]
    child1_true_params, child2_true_params = get_true_children_params(parent1.params, parent2.params, modified_beta)
    child1, child2 = DoubleParetoCross.crossover(parent1, parent2, root_individual=get_root_individual,
                                                 dpx_alpha=alpha, dpx_beta=beta)
    assert child1.params == child1_true_params
    assert child2.params == child2_true_params
