
# Standard library

# 3rd party packages

# Local source
from domain.adaptation.factory import adaptation_factory
from domain.adaptation.xiao import XiaoAdapt
from domain.adaptation.srinivas import SrinivasAdapt


def test_get_xiao_adaptation():
    xiao_adapt = adaptation_factory('xiao')
    assert xiao_adapt == XiaoAdapt


def test_get_srinivas_adaptation():
    srinivas_adapt = adaptation_factory('srinivas')
    assert srinivas_adapt == SrinivasAdapt

