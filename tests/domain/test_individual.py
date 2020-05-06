
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.domain.cost.reax_error import ReaxError


@pytest.fixture()
@pytest.mark.usefixtures('dft_energies', 'weights', 'root_ffield', 'param_keys')
def root_individual(dft_energies, weights, root_ffield, param_keys):
    root_individual = RootIndividual(dft_energies, weights, root_ffield, param_keys)
    return root_individual


@pytest.fixture()
def individual():
    params = [0.5, 0.4, 1.0, -4.7, 8.6]
    reax_energies = [43.2, 10.6, -174.2, 1008.4]
    individual = Individual(params=params, reax_energies=reax_energies, error_calculator=ReaxError)
    return individual


def test_individual_init():
    params = [0.5, 0.4, 1.0, -4.7, 8.6]
    reax_energies = [43.2, 10.6, -174.2, 1008.4]
    individual = Individual(params=params, reax_energies=reax_energies)
    assert individual.params == params
    assert individual.reax_energies == reax_energies


def test_individual_total_error(individual, root_individual):
    assert abs(individual.total_error(root_individual) - 871.9892) <= 1e-4


def test_individual_update_ffield(individual, root_individual):
    individual.ffield = individual.update_ffield(root_individual)
    assert individual.ffield != root_individual.root_ffield
    assert individual.ffield[2][0][0] == 0.5
    assert individual.ffield[2][3][0] == 0.4
    assert individual.ffield[3][0][0] == 1.0
    assert individual.ffield[3][0][2] == -4.7
    assert individual.ffield[3][2][1] == 8.6
    assert root_individual.root_ffield[2][0][0] == 1
    assert root_individual.root_ffield[2][3][0] == 7
    assert root_individual.root_ffield[3][0][0] == 1
    assert root_individual.root_ffield[3][0][2] == 3
    assert root_individual.root_ffield[3][2][1] == 8
    # Assure no other parameters are affected by the changes
    for (k1, v1), (k2, v2) in zip(individual.ffield.items(), root_individual.root_ffield.items()):
        if k1 == 1:
            assert v1 == v2
        elif k1 == 2:
            for i in range(len(v1)):
                dont_check = {0, 3}
                if i not in dont_check:
                    assert v1[i] == v2[i]
                else:
                    for j in range(len(v1[i])):
                        dont_check = {0}
                        if j not in dont_check:
                            assert v1[i][j] == v2[i][j]
        else:
            for i in range(len(v1)):
                dont_check = {0, 2}
                if i not in dont_check:
                    assert v1[i] == v2[i]
                else:
                    for j in range(len(v1[i])):
                        dont_check = {0, 1, 2}
                        if j not in dont_check:
                            assert v1[i][j] == v2[i][j]


def test_individual_from_root_individual(root_individual):
    individual_from_root = Individual.from_root_individual(root_individual)
    assert isinstance(individual_from_root, Individual)
    assert individual_from_root.params == root_individual.root_params
    assert individual_from_root.reax_energies is None
    assert individual_from_root.ffield is not None
    assert individual_from_root.cost is None


@pytest.mark.usefixtures('get_individuals')
def test_individual_eq(get_individuals):
    get_individuals[0].cost = 5142
    get_individuals[1].cost = 4387
    get_individuals[2].cost = 5142
    assert get_individuals[0] == get_individuals[2]
    assert get_individuals[0] != get_individuals[1]


@pytest.mark.usefixtures('get_individuals')
def test_individual_lt(get_individuals):
    get_individuals[0].cost = 5142
    get_individuals[1].cost = 4387
    assert get_individuals[1] < get_individuals[0]


@pytest.mark.usefixtures('get_individuals')
def test_individual_gt(get_individuals):
    get_individuals[0].cost = 5142
    get_individuals[1].cost = 4387
    assert get_individuals[0] > get_individuals[1]
