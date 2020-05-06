#!/usr/bin/env python

"""Factory for adaptation algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from domain.adaptation.srinivas import SrinivasAdapt
from domain.adaptation.xiao import XiaoAdapt


def adaptation_factory(algorithm_name: str):
    """Factory to select adaptation type."""
    adaptation_types = {
        'srinivas': SrinivasAdapt,
        'xiao': XiaoAdapt,
    }
    return adaptation_types[algorithm_name]
