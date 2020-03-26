
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.cost.reax_error import ReaxError


@pytest.mark.usefixtures('reax_energies', 'dft_energies', 'weights')
def test_reax_error(reax_energies, dft_energies, weights):
    reax_energy = reax_energies[0][0]
    dft_energy = dft_energies[0]
    weight = weights[0]
    true_error = ((reax_energy - dft_energy) / weight) ** 2
    assert ReaxError.error(reax_energy, dft_energy, weight) == true_error
    assert reax_energies[0][0] == reax_energy
    assert dft_energies[0] == dft_energy
    assert weights[0] == weight
