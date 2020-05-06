
# Standard library
import os

# 3rd party packages
import pytest

# Local source
from parametrization_clean.infrastructure.utils.reax_reader import ReaxReader, write_ffield, read_array
from parametrization_clean.domain.utils.helpers import get_param


@pytest.fixture(autouse=True)
@pytest.mark.usefixtures("reax_output_dir_path")
def module_setup_teardown(reax_output_dir_path):
    # Setup
    yield
    # Teardown
    if os.path.isfile(os.path.join(reax_output_dir_path, 'ffield')):
        os.remove(os.path.join(reax_output_dir_path, 'ffield'))


@pytest.fixture()
@pytest.mark.usefixtures("training_set_dir_path")
def reax_io_obj(training_set_dir_path):
    return ReaxReader(training_set_dir_path)


@pytest.mark.usefixtures("training_set_dir_path")
def test_reax_io_init(reax_io_obj, training_set_dir_path):
    assert reax_io_obj.dir_path == training_set_dir_path


def test_reax_io_read_ffield(reax_io_obj):
    ffield, atom_types = reax_io_obj.read_ffield()
    assert isinstance(ffield, dict)
    assert isinstance(atom_types, dict)
    assert get_param([2, 14, 1], ffield) == 1.8862
    assert get_param([2, 14, 25], ffield) == -3.0614
    assert get_param([3, 38, 4], ffield) == -0.6944
    assert get_param([4, 19, 1], ffield) == 0.0987
    assert get_param([5, 94, 5], ffield) == 2.0000
    assert atom_types[3][37 - 1][0] == '3'
    assert atom_types[3][37 - 1][1] == '14'


def test_reax_io_read_fort99(reax_io_obj):
    fort99_results = reax_io_obj.read_fort99()
    assert fort99_results[0][1] == 1.0950
    assert fort99_results[62][1] == 107.2990
    assert fort99_results[65][2] == 0.0500
    assert fort99_results[-1][0] == -0.7802


def test_reax_io_read_params(reax_io_obj):
    param_keys, param_increments, param_bounds = reax_io_obj.read_params()
    assert param_keys[0] == [2, 14, 1]
    assert param_increments[0] == 0.001
    assert param_bounds[0] == [1.8, 2.5]
    assert param_keys[22] == [4, 18, 1]
    assert param_increments[22] == 0.010
    assert param_bounds[22] == [0.02, 0.30]
    assert param_keys[-1] == [5, 94, 7]
    assert param_increments[-1] == 0.050
    assert param_bounds[-1] == [1.1, 4.0]


@pytest.mark.usefixtures("training_set_dir_path", "reax_output_dir_path")
def test_write_ffield(reax_io_obj, training_set_dir_path, reax_output_dir_path):
    input_path = os.path.join(training_set_dir_path, "ffield")
    output_path = os.path.join(reax_output_dir_path, "ffield")
    ffield, atom_types = reax_io_obj.read_ffield()
    write_ffield(input_path, output_path, ffield, atom_types)
    new_reax_io_obj = ReaxReader(reax_output_dir_path)
    new_ffield, new_atom_types = new_reax_io_obj.read_ffield()
    assert get_param([2, 14, 1], new_ffield) == 1.8862
    assert get_param([2, 14, 25], new_ffield) == -3.0614
    assert get_param([3, 38, 4], new_ffield) == -0.6944
    assert get_param([4, 19, 1], new_ffield) == 0.0987
    assert get_param([5, 94, 5], new_ffield) == 2.0000
    assert new_atom_types[3][37 - 1][0] == '3'
    assert new_atom_types[3][37 - 1][1] == '14'
