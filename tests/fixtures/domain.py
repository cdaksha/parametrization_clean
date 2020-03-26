# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.individual import Individual


@pytest.fixture()
def get_individuals():
    individuals = [Individual(params=[1.0, 0.1, -0.5, 4.3, 2.4],
                              reax_energies=[43.2, 10.6, -174.2, 1008.4]),
                   Individual(params=[0.5, 0.05, -0.76, 8.6, 2.1],
                              reax_energies=[56.9, 11.2, -164.1, 994.8]),
                   Individual(params=[1.24, 0.17, -0.07, 6.3, 2.24],
                              reax_energies=[102.4, 5.5, -155.5, 1008.12]),
                   Individual(params=[0.73, 0.25, -0.25, 10.2, 1.7],
                              reax_energies=[40.2, 11.2, -104.1, 1018.32]),
                   ]
    return individuals


@pytest.fixture()
def reax_energies():
    reax_energies = [[43.2, 10.6, -174.2, 1008.4],
                     [56.9, 11.2, -164.1, 994.8],
                     [102.4, 5.5, -155.5, 1008.12],
                     [40.2, 11.2, -104.1, 1018.32],
                     ]
    return reax_energies


@pytest.fixture()
def dft_energies():
    return [42.1, 9.8, -154.3, 980.6]


@pytest.fixture()
def weights():
    return [4.0, 3.0, 2.0, 1.0]


@pytest.fixture()
def root_ffield():
    ffield = {1: [1, 2, 3, 4, 5],
              2: [[1, 2], [3, 4], [5, 6], [7, 8]],
              3: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
              }
    return ffield


@pytest.fixture()
def param_keys():
    param_keys = [[2, 1, 1], [2, 4, 1], [3, 1, 1], [3, 1, 3], [3, 3, 2]]
    return param_keys


@pytest.fixture()
def param_bounds():
    return [[0.0, 2.0], [0.0, 0.5], [-1.0, 0.0], [3.0, 12.0], [1.0, 3.0]]
