#!/usr/bin/env python

"""Module with interface for computation of objective function."""

# Standard library
import abc

# 3rd party packages

# Local source


class IErrorStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def error(reax_val, dft_val, weight, **kwargs) -> float:
        raise NotImplementedError
