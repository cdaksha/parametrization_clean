
# Standard library
from copy import deepcopy

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.utils.helpers import set_param, get_param


@pytest.mark.usefixtures('get_individuals', 'root_ffield', 'param_keys')
def test_set_param(get_individuals, root_ffield, param_keys):
    param = get_individuals[2].params[0]
    original_ffield = deepcopy(root_ffield)
    key = param_keys[0]
    set_param(key, root_ffield, param)
    assert root_ffield[2][0][0] == 1.24
    assert root_ffield[2][0][0] != original_ffield[2][0][0]
    # Assure no other parameters are affected by the single change
    for (k1, v1), (k2, v2) in zip(root_ffield.items(), original_ffield.items()):
        if k1 != 2:
            assert v1 == v2
        else:
            for val1, val2 in zip(v1[1:], v2[1:]):
                assert val1 == val2


@pytest.mark.usefixtures('root_ffield', 'param_keys')
def test_get_param(root_ffield, param_keys):
    key = param_keys[0]
    param = get_param(key, root_ffield)
    assert param == root_ffield[2][0][0]
    assert param == 1
