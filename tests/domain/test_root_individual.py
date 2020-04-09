
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.root_individual import RootIndividual


@pytest.mark.usefixtures('dft_energies', 'weights', 'root_ffield', 'param_keys')
def test_root_individual_init(dft_energies, weights, root_ffield, param_keys):
    root_individual = RootIndividual(dft_energies, weights, root_ffield, param_keys)
    true_params = [1, 7, 1, 3, 8]
    assert list(root_individual.dft_energies) == dft_energies
    assert list(root_individual.weights) == weights
    assert root_individual.root_ffield == root_ffield
    assert root_individual.param_keys == param_keys
    assert root_individual.extract_params() == true_params
    assert root_individual.root_params == true_params
