#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that contains mutation methods to be used for a Child.

__author__ = "Chad Daksha"
"""

# Standard library
import random
import math

# 3rd party packages
import numpy as np

# Local source
import core.genetic_algorithm.individual as c
from core.settings.config import settings as s
from core.genetic_algorithm.helpers import mutate_param

FRAC_PARAMS_MUTATE = s.GA.mutation.fractionParametersToMutate
if FRAC_PARAMS_MUTATE == 'all':
    FRAC_PARAMS_MUTATE = 1.0
elif FRAC_PARAMS_MUTATE == 'random':
    FRAC_PARAMS_MUTATE = random.random()


def get_mutation(mutation_type: str = 'polynomial'):
    """Factory to select mutation type. Default is polynomial mutation. Note that Nakata mutation is present in
    c.Child class.
    """
    mutation_types = {
        'polynomial': polynomial_mutate,
        'polynomial_no_bounds': polynomial_no_bounds_mutate,
        'central_uniform': central_uniform_mutate,  # Initialization Algorithm
        'nakata': nakata_mutate,  # Can be used as an Initialization Algorithm
        'gauss': gaussian_mutate,
        'multi_gauss': multi_gauss_mutate,
        'log_logistic': log_logistic_mutate,
        'gauss_and_cauchy': gauss_and_cauchy_mutate,
        'AMMO': ammo_mutate,
    }
    return mutation_types[mutation_type]


def polynomial_mutate(parent: c.Individual) -> c.Individual:
    """Polynomial mutation according to Deb and Agrawal's paper."""
    polynomial_eta = s.AlgorithmParameters.mutation.polynomial.eta
    num_mutate = int(FRAC_PARAMS_MUTATE * len(parent.params))

    new_params = [param for param in parent.params]  # Copying over to prevent changing original params
    poly_degree = 1 / (1 + polynomial_eta)

    idx = random.sample(range(len(parent.params)), num_mutate)
    for i in idx:
        param = new_params[i]
        lower_bound = parent.param_bounds[i][0]
        upper_bound = parent.param_bounds[i][1]

        u = random.random()
        if u <= 0.5:
            delta_l = (2 * u) ** poly_degree - 1
            param = param + delta_l * (param - lower_bound)
        else:
            delta_r = 1 - (2 * (1 - u)) ** poly_degree
            param = param + delta_r * (upper_bound - param)

        # Ensuring new parameter is within bounds
        param = min(max(param, lower_bound), upper_bound)

        new_params[i] = param

    return c.Individual(new_params)


def polynomial_no_bounds_mutate(parent: c.Individual) -> c.Individual:
    """Polynomial mutation adapted from Deb and Agrawal's paper.
    ADAPTED TO FUNCTION WITHOUT UPPER/LOWER BOUNDS FOR PARAMETERS.
    """
    polynomial_eta = s.AlgorithmParameters.mutation.polynomial.eta
    num_mutate = int(FRAC_PARAMS_MUTATE * len(parent.params))

    new_params = [param for param in parent.params]  # Copying over to prevent changing original params
    poly_degree = 1 / (1 + polynomial_eta)

    idx = random.sample(range(len(parent.params)), num_mutate)
    for i in idx:
        param = new_params[i]

        u = random.random()
        if u <= 0.5:
            delta_l = (2 * u) ** poly_degree - 1
            param = param + delta_l * param
        else:
            delta_r = 1 - (2 * (1 - u)) ** poly_degree
            param = param + delta_r * param

        new_params[i] = param

    return c.Individual(new_params)


def central_uniform_mutate(parent: c.Individual) -> c.Individual:
    """Inspired by Monte Carlo/GA guidelines for ReaxFF paper.
    Use a random number (determined by uniform distribution) in the central segment for each parameter range.
    So, if a parameter has a range of [p_min, p_max], a uniform random number will be generated in the bounds
    [p_min + (p_max - p_min)/4, p_max - (p_max - p_min)/4].
    Mutates all parameters (doesn't consider `FRAC_PARAMS_MUTATE`).
    """
    new_params = []
    for (lower_bound, upper_bound) in parent.param_bounds:
        delta = (upper_bound - lower_bound) / 4
        new_param = random.uniform(lower_bound + delta, upper_bound - delta)
        new_params.append(new_param)
    return c.Individual(new_params)


def nakata_mutate(parent: c.Individual) -> c.Individual:
    """Mutate Child's `params` using Nakata's methodology. More details are in `mutate_param`.

    `scale` can be float or List with len(`scale`) = len(self.params), e.g. when using param_increments is desired.
    param_bounds are currently being used as (min, max) conditions for params, if they exist.
    if param is outside param_bounds, param is set using uniform distribution with (min, max) bounds.

    Returns
    -------
    New Child after mutation.
    """
    scale = s.AlgorithmParameters.mutation.nakata.scale
    low = s.AlgorithmParameters.mutation.nakata.randomLower
    high = s.AlgorithmParameters.mutation.nakata.randomUpper
    num_mutate = int(FRAC_PARAMS_MUTATE * len(parent.params))

    if scale == 'param_increments':
        scale_list = parent.param_increments
    else:
        scale_list = [scale] * len(parent.params)

    new_params = [param for param in parent.params]  # Copying over to prevent changing original params
    idx = random.sample(range(len(parent.params)), num_mutate)
    for i in idx:
        scale_factor = scale_list[i]
        new_param = mutate_param(parent.params[i], scale_factor, low, high)

        # Comment out for now because trying to parametrize without bounds
        # if parent.param_bounds[i]:
        #     lower_bound = parent.param_bounds[i][0]
        #     upper_bound = parent.param_bounds[i][1]
        #     if not lower_bound <= new_param <= upper_bound:
        #         new_param = random.uniform(lower_bound, upper_bound)

        new_params[i] = new_param

    return c.Individual(new_params)


def gaussian_mutate(parent: c.Individual) -> c.Individual:
    """UNCONSTRAINED standard normal Gaussian mutation - N(0, 1)
    Upper and lower bounds are not required for this version.
    """
    mu = 0
    std = s.AlgorithmParameters.mutation.gauss_and_cauchy.sigma_gauss
    new_params = [param + param * random.gauss(mu, std) for param in parent.params]
    return c.Individual(new_params)


def cauchy_mutate(parent: c.Individual) -> c.Individual:
    """UNCONSTRAINED standard normal cauchy mutation - C(0, 1)
    Upper and lower bounds are not required.
    """
    new_params = [param + param * np.random.standard_cauchy() for param in parent.params]
    return c.Individual(new_params)


def gauss_and_cauchy_mutate(parent: c.Individual) -> c.Individual:
    """Draw a random number with equal probability to use Gaussian and Cauchy mutation schemes."""
    u = random.uniform(0, 1)
    if u < 0.5:
        child = gaussian_mutate(parent)
    else:
        child = cauchy_mutate(parent)
    return child


def multi_gauss_mutate(parent: c.Individual) -> c.Individual:
    """Uses multiple scaling factors for the normal distribution for mutation."""
    std_factors = s.AlgorithmParameters.mutation.multi_gauss.std
    std_probabilities = s.AlgorithmParameters.mutation.multi_gauss.frac
    u = random.uniform(0, 1)
    cumulative_probability = 0
    for std, probability in zip(std_factors, std_probabilities):
        if cumulative_probability <= u <= (cumulative_probability + probability):
            print("{fmt}MULTI GAUSS MUTATION{fmt}".format(fmt='-'*50))
            print("Std Factor: {}".format(std))
            new_params = [param + param * random.gauss(0, std) for param in parent.params]
            break
        cumulative_probability += probability
    print("Old Params: {}".format(parent.params))
    print("New Params: {}".format(new_params))
    return c.Individual(new_params)


def log_logistic_mutate(parent: c.Individual) -> c.Individual:
    """Log logistic mutation based on Deep 2012."""
    alpha = s.AlgorithmParameters.mutation.log_logistic.alpha
    beta = 1
    new_params = []
    for param in parent.params:
        h = random.uniform(0, 1)
        log_logistic_rand_num = beta * (h / (1 - h)) ** (1 / alpha)
        new_param = param + param * log_logistic_rand_num
        new_params.append(new_param)
    print("Old Params: {}".format(parent.params))
    print("New Params: {}".format(new_params))
    return c.Individual(new_params)


def ammo_mutate(parent: c.Individual) -> c.Individual:
    """Adaptive Mean Mutation Operator from Chellapilla's 1998 paper, "Combining Mutation Operators in
    Evolutionary Programming".
    Note: Mutates ALL parameters by default! (as done by original authors)
    NOTE: AMMO Mutate doesn't work for this case! Need to keep track of new vs. old strategy parameters
    and change them accordingly as generations go on, which is not what the current GA is about.
    """
    tau = 1 / math.sqrt(2 * len(parent.params))
    tau_prime = 1 / math.sqrt(2 * math.sqrt(len(parent.params)))

    individual_normal_val = random.gauss(0, 1)
    individual_normal_val_2 = random.gauss(0, 1)
    individual_normal_vals = [random.gauss(0, 1) for _ in range(len(parent.params))]
    individual_normal_vals_2 = [random.gauss(0, 1) for _ in range(len(parent.params))]
    individual_cauchy_vals = np.random.standard_cauchy(len(parent.params))

    sigma_primes_1 = [max(s.AlgorithmParameters.mutation.AMMO.minimum_sigma,
                          s.AlgorithmParameters.mutation.AMMO.initial_sigma *
                          math.exp(tau_prime * val + tau * individual_normal_val))
                      for val in individual_normal_vals]
    sigma_primes_2 = [max(s.AlgorithmParameters.mutation.AMMO.minimum_sigma,
                          s.AlgorithmParameters.mutation.AMMO.initial_sigma *
                          math.exp(tau_prime * val + tau * individual_normal_val_2))
                      for val in individual_normal_vals_2]

    new_params = [param + sigma1 * ind_normal_val + sigma2 * ind_cauchy_val
                  for param, sigma1, sigma2, ind_normal_val, ind_cauchy_val in
                  zip(parent.params, sigma_primes_1, sigma_primes_2, individual_normal_vals, individual_cauchy_vals)]

    return c.Individual(new_params)
