#!/usr/bin/env python

# Standard library
from typing import List
import random

# 3rd party packages


# Local source
from parametrization_clean.domain.selection.strategy import ISelectionStrategy
from parametrization_clean.domain.individual import Individual


class TournamentSelect(ISelectionStrategy):

    @staticmethod
    def selection(population: List[Individual], **kwargs) -> Individual:
        """Select `tournament_size` individuals from `population` at random and return best individual by fitness.
        Default tournament size = 2.
        """
        tournament_size = kwargs.get('tournament_size', 2)
        selected = random.sample(population, tournament_size)
        return min(selected)
