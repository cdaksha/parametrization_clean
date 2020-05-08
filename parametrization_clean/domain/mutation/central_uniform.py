#!/usr/bin/env python

# Standard library
import random

# 3rd party packages


# Local source
from parametrization_clean.domain.mutation.strategy import IMutationStrategy
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual


class CentralUniformMutate(IMutationStrategy):

    @staticmethod
    def mutation(parent: Individual, root_individual: RootIndividual, **kwargs) -> Individual:
        """Inspired by Monte Carlo/GA guidelines for ReaxFF paper.
        Use a random number (determined by uniform distribution) in the central segment for each parameter range.
        So, if a parameter has a range of [p_min, p_max], a uniform random number will be generated in the bounds
        [p_min + (p_max - p_min)/4, p_max - (p_max - p_min)/4].
        Mutates all parameters (doesn't consider `FRAC_PARAMS_MUTATE`).
        """
        param_bounds = kwargs['param_bounds']
        new_params = []
        for (lower_bound, upper_bound) in param_bounds:
            delta = (upper_bound - lower_bound) / 4
            new_param = random.uniform(lower_bound + delta, upper_bound - delta)
            new_params.append(new_param)
        return Individual(new_params, root_individual=root_individual)
