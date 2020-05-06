#!/usr/bin/env python

"""Factory for selection algorithms allowed for usage."""

# Standard library

# 3rd party packages

# Local source
from domain.selection.tournament import TournamentSelect


def selection_factory(algorithm_name: str):
    """Factory to select selection type."""
    selection_types = {
        'tournament': TournamentSelect
    }
    return selection_types[algorithm_name]
