
# Standard library
from unittest.mock import patch

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.mutation.gauss import GaussianMutate
from tests.domain.crossover.test_double_pareto import get_root_individual


@pytest.mark.usefixtures('get_individuals')
def test_single_gauss_with_frac(get_root_individual, get_individuals):
    parent = get_individuals[0]
    child = GaussianMutate.mutation(parent, get_root_individual)
    assert parent.params != child.params
    assert parent.params == [1.0, 0.1, -0.5, 4.3, 2.4]


@pytest.mark.usefixtures('get_individuals')
def test_multi_gauss(get_root_individual, get_individuals):
    parent = get_individuals[0]
    child = GaussianMutate.mutation(parent, root_individual=get_root_individual,
                                    gauss_std=[0.01, 0.1, 1.0], gauss_frac=[0.25, 0.5, 0.25])
    assert parent.params != child.params
    assert parent.params == get_individuals[0].params


@patch('random.gauss')
@patch('random.uniform')
@pytest.mark.usefixtures('get_individuals')
def test_multi_gauss_std_1(uniform_mock, gauss_mock, get_root_individual, get_individuals):
    uniform_mock.return_value = 0.15
    parent = get_individuals[0]
    GaussianMutate.mutation(parent, root_individual=get_root_individual,
                            gauss_std=[0.01, 0.1, 1.0], gauss_frac=[0.25, 0.5, 0.25])
    gauss_mock.assert_called_with(0, 0.01)


@patch('random.gauss')
@patch('random.uniform')
@pytest.mark.usefixtures('get_individuals')
def test_multi_gauss_std_2(uniform_mock, gauss_mock, get_root_individual, get_individuals):
    uniform_mock.return_value = 0.50
    parent = get_individuals[0]
    GaussianMutate.mutation(parent, root_individual=get_root_individual,
                            gauss_std=[0.01, 0.1, 1.0], gauss_frac=[0.25, 0.5, 0.25])
    gauss_mock.assert_called_with(0, 0.1)


@patch('random.gauss')
@patch('random.uniform')
@pytest.mark.usefixtures('get_individuals')
def test_multi_gauss_std_1(uniform_mock, gauss_mock, get_root_individual, get_individuals):
    uniform_mock.return_value = 0.85
    parent = get_individuals[0]
    GaussianMutate.mutation(parent, root_individual=get_root_individual,
                            gauss_std=[0.01, 0.1, 1.0], gauss_frac=[0.25, 0.5, 0.25])
    gauss_mock.assert_called_with(0, 1.0)
