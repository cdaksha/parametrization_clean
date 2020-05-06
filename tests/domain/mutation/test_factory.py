
# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.mutation.factory import mutation_factory
from parametrization_clean.domain.mutation.nakata import NakataMutate
from parametrization_clean.domain.mutation.central_uniform import CentralUniformMutate
from parametrization_clean.domain.mutation.polynomial import PolynomialMutate
from parametrization_clean.domain.mutation.gauss import GaussianMutate


def test_get_nakata():
    assert mutation_factory('nakata') == NakataMutate


def test_get_central_uniform():
    assert mutation_factory('central_uniform') == CentralUniformMutate


def test_get_polynomial():
    assert mutation_factory('polynomial') == PolynomialMutate


def test_get_gauss():
    assert mutation_factory('gauss') == GaussianMutate
