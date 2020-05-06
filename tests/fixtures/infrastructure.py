
# Standard library
import os

# 3rd party packages
import pytest

# Local source


@pytest.fixture()
def training_set_dir_path():
    tests_dir = os.path.join(os.getcwd(), "tests")
    return os.path.join(tests_dir, "integration", "reference_training_set",
                        "ReaxFF_ZnO_Raymand_with_Sglass_control_all_bounds")


@pytest.fixture()
def reax_output_dir_path():
    tests_dir = os.path.join(os.getcwd(), "tests")
    return os.path.join(tests_dir, "integration", "reax_outputs")


@pytest.fixture()
def presenter_output_path():
    tests_dir = os.path.join(os.getcwd(), "tests")
    return os.path.join(tests_dir, "integration", "presenter")


@pytest.fixture()
def cli_output_path():
    tests_dir = os.path.join(os.getcwd(), "tests")
    return os.path.join(tests_dir, "integration", "cli")
