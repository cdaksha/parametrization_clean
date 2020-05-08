
# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.cost.factory import ErrorFactory
from parametrization_clean.domain.cost.reax_error import ReaxError


def test_get_reax_error():
    reax_error_calculator = ErrorFactory.create_executor('reax_error')
    assert reax_error_calculator == ReaxError

