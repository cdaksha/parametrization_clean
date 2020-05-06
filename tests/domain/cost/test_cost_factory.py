
# Standard library

# 3rd party packages

# Local source
from domain.cost.factory import error_calculator_factory
from domain.cost.reax_error import ReaxError


def test_get_reax_error():
    reax_error_calculator = error_calculator_factory('reax_error')
    assert reax_error_calculator == ReaxError

