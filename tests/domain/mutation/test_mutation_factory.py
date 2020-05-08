
# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.mutation.factory import MutationFactory
from parametrization_clean.domain.mutation.nakata import NakataMutate
from parametrization_clean.domain.mutation.central_uniform import CentralUniformMutate
from parametrization_clean.domain.mutation.polynomial import PolynomialMutate
from parametrization_clean.domain.mutation.gauss import GaussianMutate


def test_get_nakata():
    assert MutationFactory.create_executor('nakata') == NakataMutate


def test_get_central_uniform():
    assert MutationFactory.create_executor('central_uniform') == CentralUniformMutate


def test_get_polynomial():
    assert MutationFactory.create_executor('polynomial') == PolynomialMutate


def test_get_gauss():
    assert MutationFactory.create_executor('gauss') == GaussianMutate
