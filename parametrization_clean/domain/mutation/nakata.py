#!/usr/bin/env python

# Standard library
import random

# 3rd party packages


# Local source
from parametrization_clean.domain.mutation.strategy import IMutationStrategy
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual


class NakataMutate(IMutationStrategy):

    @staticmethod
    def mutation(parent: Individual, root_individual: RootIndividual, **kwargs) -> Individual:
        """Mutate Child's `params` using Nakata's methodology.
        new_param = old_param + (scale * rand_num * old_param).

        `scale` can be float or List with len(`scale`) = len(self.params),
        e.g., when using param_increments is desired.
        param_bounds are currently being used as (min, max) conditions for params, if they exist.
        if param is outside param_bounds, param is set using uniform distribution with (min, max) bounds.

        :return: New Individual after mutation.
        """
        scale = kwargs.get('nakata_scale', 0.1)
        low = kwargs.get('nakata_rand_lower', -1.0)
        high = kwargs.get('nakata_rand_higher', 1.0)
        new_params = [param + (scale * random.uniform(low, high) * param) for param in parent.params]
        return Individual(new_params, root_individual=root_individual)
