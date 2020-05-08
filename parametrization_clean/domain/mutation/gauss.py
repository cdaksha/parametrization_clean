#!/usr/bin/env python

# Standard library
import random

# 3rd party packages


# Local source
from parametrization_clean.domain.mutation.strategy import IMutationStrategy
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual


class GaussianMutate(IMutationStrategy):

    @staticmethod
    def mutation(parent: Individual, root_individual: RootIndividual, **kwargs) -> Individual:
        """UNCONSTRAINED Gaussian mutation.
        Upper and lower bounds are not required for this version.
        Allows usage of multiple scaling factors for the normal distribution for mutation.
        Each scaling factor needs a corresponding probability (gauss_frac) of using that given factor.
        """
        stds = kwargs.get('gauss_std', [0.1])
        probabilities = kwargs.get('gauss_frac', [1.0])
        u = random.uniform(0, 1)
        cumulative_probability = 0
        for std, probability in zip(stds, probabilities):
            if cumulative_probability < u <= (cumulative_probability + probability):
                new_params = [param + param * random.gauss(0, std) for param in parent.params]
                break
            cumulative_probability += probability
        return Individual(new_params, root_individual=root_individual)
