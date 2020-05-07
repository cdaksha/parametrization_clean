#!/usr/bin/env python

"""Module with standard ReaxFF error function implementation."""

# Standard library

# 3rd party packages

# Local source
from domain.cost.strategy import IErrorStrategy
from domain.cost.factory import ErrorFactory


@ErrorFactory.register('reax_error')
class ReaxError(IErrorStrategy):

    @staticmethod
    def error(reax_val, dft_val, weight, **kwargs) -> float:
        """Calculate ReaxFF error using error = ((reax_pred - true_val)/weight)^2."""
        return ((reax_val - dft_val) / weight) ** 2
