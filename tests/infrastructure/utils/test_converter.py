
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.infrastructure.utils.reax_reader import ReaxReader
from parametrization_clean.infrastructure.utils.reax_converter import Fort99Extractor


@pytest.fixture()
@pytest.mark.usefixtures("training_set_dir_path")
def fort99_data(training_set_dir_path):
    reax_reader = ReaxReader(training_set_dir_path)
    return reax_reader.read_fort99()


@pytest.fixture()
def fort99_extractor(fort99_data):
    return Fort99Extractor(fort99_data)


def test_get_reax_energies(fort99_extractor):
    reax_energies = fort99_extractor.get_reax_energies()
    assert isinstance(reax_energies, list)
    assert reax_energies[0] == 1.1183
    assert reax_energies[-1] == -0.7802
    assert reax_energies[180] == 1.4460


def test_get_dft_energies(fort99_extractor):
    dft_energies = fort99_extractor.get_dft_energies()
    assert isinstance(dft_energies, list)
    assert dft_energies[0] == 1.0950
    assert dft_energies[-1] == -0.0010
    assert dft_energies[180] == 0.1090


def test_get_weights(fort99_extractor):
    weights = fort99_extractor.get_weights()
    assert isinstance(weights, list)
    assert weights[0] == 0.0500
    assert weights[-1] == 2.0000
    assert weights[180] == 1.0000
