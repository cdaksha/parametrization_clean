
# Standard library

# 3rd party packages

# Local source
from domain.adaptation.factory import AdaptationFactory
from domain.adaptation.xiao import XiaoAdapt
from domain.adaptation.srinivas import SrinivasAdapt


def test_get_xiao_adaptation():
    xiao_adapt = AdaptationFactory.create_executor('xiao')
    assert xiao_adapt == XiaoAdapt


def test_get_srinivas_adaptation():
    srinivas_adapt = AdaptationFactory.create_executor('srinivas')
    assert srinivas_adapt == SrinivasAdapt

