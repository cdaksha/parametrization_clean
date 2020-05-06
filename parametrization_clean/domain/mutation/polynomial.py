#!/usr/bin/env python

# Standard library
import random

# 3rd party packages


# Local source
from parametrization_clean.domain.mutation.strategy import IMutationStrategy
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual


class PolynomialMutate(IMutationStrategy):

    @staticmethod
    def mutation(parent: Individual, root_individual: RootIndividual, **kwargs) -> Individual:
        """Polynomial mutation adapted from Deb and Agrawal's paper.
        ADAPTED TO FUNCTION WITHOUT UPPER/LOWER BOUNDS FOR PARAMETERS.
        """
        eta = kwargs.get('polynomial_eta', 60)
        poly_degree = 1 / (1 + eta)
        new_params = []
        for param in parent.params:
            u = random.random()
            if u <= 0.5:
                delta_l = (2 * u) ** poly_degree - 1
                param = param + delta_l * param
            else:
                delta_r = 1 - (2 * (1 - u)) ** poly_degree
                param = param + delta_r * param
            new_params.append(param)
        return Individual(new_params, root_individual=root_individual)
