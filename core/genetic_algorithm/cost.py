#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module with class to structure/maintain genetic_algorithm Individual's fitness.

__author__ = "Chad Daksha"
"""

# Standard library
import statistics


class Cost(object):

    def __init__(self, weights):
        """Computes the Cost for an Individual in the genetic algorithm.
        As opposed to Fitness, a lower Cost is better than a higher Cost.

        Parameters
        ----------
        weights: List[float]
            list containing weights for each parameter being trained.
        errors: List[float]
            list containing reaxFF errors.
        """
        self.weights = weights

    @staticmethod
    def total_error(errors) -> float:
        """Sum of all errors."""
        return sum(errors)

    def error_for_important_lines(self, errors, weight_threshold=1.0) -> float:
        """Sum all errors for lines with weights less than or equal to `weight_threshold`."""
        relevant_errors = self.get_errors_for_important_lines(errors, weight_threshold)
        return sum(relevant_errors)

    def get_errors_for_important_lines(self, errors, weight_threshold=1.0):
        """Return a list of errors that have weights less than or equal to `weight_threshold`."""
        return [error for error, weight in zip(errors, self.weights) if weight <= weight_threshold]


class PopulationCost(object):

    def __init__(self, individuals):
        """Class to compute cost metric for each individual for a list of individuals."""
        self.individuals = individuals
        self.cost_calculator = Cost(individuals[0].weights)

        self.mean_score = 75 / 100  # assign 75/100 to be the mean normalized score
        self.score_per_std = 25 / 100  # add (or subtract) 25/100 per standard deviation away from the mean
        self.weight_threshold = 1.0  # weight threshold to calculate total error for important lines

    def get_costs(self):
        """This function serves as a high-level factory method to pick the cost function one wishes to use.
        For example, you can use just the total error, or maybe you can use the composite cost index.
        """
        # Currently testing using only the total cost
        return self._total_costs()

    def assign_costs(self, costs):
        """Assign the cost to each individual in the population."""
        for individual, cost in zip(self.individuals, costs):
            individual.cost = cost

    @staticmethod
    def _z_scores(costs):
        """Return z-scores for each individual in the population for the provided costs."""
        avg_cost = statistics.mean(costs)
        std_cost = statistics.stdev(costs)
        z_scores = [(cost - avg_cost) / std_cost for cost in costs]
        return z_scores

    def _normalized_costs(self, z_scores):
        return [self.mean_score + z_score * self.score_per_std for z_score in z_scores]

    def _total_costs(self):
        """Return total error/cost for each individual in the population."""
        return [self.cost_calculator.total_error(individual.errors) for individual in self.individuals]

    def _normalized_total_costs(self):
        """Return normalized score for each individual based on their total error; the lower, the better."""
        costs = self._total_costs()
        # Debugging
        # print("{fmt}TOTAL ERRORS{fmt}".format(fmt='-'*10))
        # print(costs)
        # print("{fmt}TOTAL ERRORS Z SCORES{fmt}".format(fmt='-' * 10))
        z_scores = self._z_scores(costs)
        # print(z_scores)
        return self._normalized_costs(z_scores)

    def _normalized_important_costs(self):
        """Return normalized score for each individual based on their error below the weight threshold."""
        costs = [self.cost_calculator.error_for_important_lines(individual.errors, self.weight_threshold)
                 for individual in self.individuals]
        # Debugging
        # print("{fmt}IMPORTANT ERRORS{fmt}".format(fmt='-' * 10))
        # print(costs)
        # print("{fmt}IMPORTANT ERRORS Z SCORES{fmt}".format(fmt='-' * 10))
        z_scores = self._z_scores(costs)
        # print(z_scores)
        return self._normalized_costs(z_scores)

    def _normalized_energy_curve_costs(self):
        """Return normalized score for each individual based on their energy curve match criteria."""
        return [get_energy_curve_cost(individual) for individual in self.individuals]

    def get_composite_costs(self):
        """Total of different costs for an overall evaluation of an Individual.
        Minimum cost ~ 0.0 (best case), Maximum cost ~ 3.0 (3 different normalized criteria).
        """
        normalized_total_costs = self._normalized_total_costs()
        normalized_energy_curve_costs = self._normalized_energy_curve_costs()
        normalized_important_costs = self._normalized_important_costs()
        # return [a + b + c for a, b, c in
        #         zip(normalized_total_costs, normalized_energy_curve_costs, normalized_important_costs)]
        # Debugging purposes
        composite_costs = []
        for a, b, c in zip(normalized_total_costs, normalized_important_costs, normalized_energy_curve_costs):
            print("Total Cost: {}\tImportant Cost: {}\tEnergy Curve Cost: {}".format(a, b, c))
            composite_costs.append(a + b + c)
        return composite_costs


# HELPERS
# Determine monotonicity
def strictly_increasing(arr):
    return all(x < y for x, y in zip(arr, arr[1:]))


def non_decreasing(arr):
    return all(x <= y for x, y in zip(arr, arr[1:]))


def non_increasing(arr):
    return all(x >= y for x, y in zip(arr, arr[1:]))
# Determine Monotonicity


# Energy Curve Matching Checks
def split_energy_curve_data(energy_curve_data):
    return ([[data[0] for data in energy_curve] for energy_curve in energy_curve_data],
            [[data[1] for data in energy_curve] for energy_curve in energy_curve_data])


def get_energy_diff(reaxff_curve_data, dft_curve_data):
    return [[reaxFF_val - DFT_val for reaxFF_val, DFT_val in zip(reaxFF_curve, DFT_curve)]
            for reaxFF_curve, DFT_curve in zip(reaxff_curve_data, dft_curve_data)]


def match_curve_minimums(reaxff_curve_data, dft_curve_data):
    total_possible_matches = len(reaxff_curve_data)
    # 0 = match found; 50 = match not found
    matches = [0 if reaxFF_curve.index(min(reaxFF_curve)) == dft_curve.index(min(dft_curve)) else 50
               for reaxFF_curve, dft_curve in zip(reaxff_curve_data, dft_curve_data)]
    print("{fmt}ENERGY MINIMUM LOCATION MATCHES{fmt}".format(fmt='-'*10))
    print(matches)
    return sum(matches) / total_possible_matches


def match_curve_monotonicitys(reaxff_curve_data, dft_curve_data):
    total_possible_matches = len(reaxff_curve_data)
    energy_difference_data = get_energy_diff(reaxff_curve_data, dft_curve_data)
    abs_energy_difference_data = [[abs(val) for val in energy_difference_curve]
                                  for energy_difference_curve in energy_difference_data]
    matches = []
    for abs_energy_difference_curve, reaxFF_curve in zip(abs_energy_difference_data, reaxff_curve_data):
        min_index = reaxFF_curve.index(min(reaxFF_curve))
        if (non_increasing(abs_energy_difference_curve[0:min_index])
                and non_decreasing(abs_energy_difference_curve[(min_index+1):])):
            match = 0
        else:
            match = 30
        matches.append(match)
    print("{fmt}ENERGY MONOTONICITY MATCHES{fmt}".format(fmt='-'*10))
    print(matches)
    return sum(matches) / total_possible_matches


def energy_curve_max_difference_matching(reaxff_curve_data, dft_curve_data):
    max_allowable_difference_fraction = 0.50  # 50% max allowed deviation from DFT value for curve endpoints
    total_possible_matches = len(reaxff_curve_data)
    energy_difference_data = get_energy_diff(reaxff_curve_data, dft_curve_data)
    # Checking all points in the curve
    matches = [20 if any(abs(energy_diff_val / dft_val) > max_allowable_difference_fraction
                         for energy_diff_val, dft_val in zip(energy_difference_curve, dft_curve)) else 0
               for energy_difference_curve, dft_curve in zip(energy_difference_data, dft_curve_data)]
    print("{fmt}ENERGY MAX DIFFERENCE MATCHES{fmt}".format(fmt='-'*10))
    print(matches)
    return sum(matches) / total_possible_matches


# Combine energy curve match criteria
def get_energy_curve_cost(individual):
    reaxff_curve_data, dft_curve_data = split_energy_curve_data(individual.energy_curve_data)
    return (match_curve_minimums(reaxff_curve_data, dft_curve_data)
            + match_curve_monotonicitys(reaxff_curve_data, dft_curve_data)
            + energy_curve_max_difference_matching(reaxff_curve_data, dft_curve_data)) / (50 + 30 + 20)
