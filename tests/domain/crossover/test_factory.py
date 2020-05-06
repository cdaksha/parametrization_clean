
# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.crossover.factory import crossover_factory
from parametrization_clean.domain.crossover.single_point import SinglePointCross
from parametrization_clean.domain.crossover.two_point import TwoPointCross
from parametrization_clean.domain.crossover.uniform import UniformCross
from parametrization_clean.domain.crossover.double_pareto import DoubleParetoCross


def test_get_single_point():
    single_point_cross = crossover_factory('single_point')
    assert single_point_cross == SinglePointCross


def test_get_two_point():
    assert crossover_factory('two_point') == TwoPointCross


def test_get_uniform():
    assert crossover_factory('uniform') == UniformCross


def test_get_double_pareto():
    assert crossover_factory('double_pareto') == DoubleParetoCross
