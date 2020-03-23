#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that contains helper methods for genetic algorithm implementation.

__author__ = "Chad Daksha"
"""

# Standard library
import random
from typing import List, Dict, Tuple
from itertools import combinations  # Find unique parent combinations for crossover

# 3rd party packages

# Local source


# ---------- HELPER FUNCTIONS ----------
# GA Algorithm Helpers
def reax_error(reax_pred, true_val, weight):
    """Calculate ReaxFF error using error = ((reax_pred - true_val)/weight)^2."""
    return ((reax_pred - true_val) / weight) ** 2


def sort_by_cost(population: List) -> List:
    """Return list with sorted Children based on the cost metric (ex. total_error)."""
    return sorted(population, key=lambda x: x.cost)


def sort_costs_and_case_numbers(population, case_numbers) -> List[Tuple[float, float]]:
    """Link a child to it's case number (i.e. child-56) then sort based on the error of the corresponding case.
    Returns a list of tuples containing (case_number, cost).
    """
    sorted_case_numbers_and_costs = sorted(zip(case_numbers, map(lambda x: x.cost, population)),
                                           key=lambda x: x[1])
    return sorted_case_numbers_and_costs


def get_best_child(population: List):
    """Return best Child from population based on the cost metric (ex. total_error)."""
    return sort_by_cost(population)[0]


def mutate_param(param: float, scale: float, low: float, high: float) -> float:
    """Based on Nakata's methodology: Given one parameter, transform it by using
    new_param = old_param + (scale * old_param * rand_num).

    Parameters
    ----------
    param: float
    scale: float
    low: float
        Lower bound for random number
    high: float
        Upper bound for random number

    Returns
    -------
    new_param: float
        Scaled factor based on Nakata's methodology.
    """
    return param + (scale * param * random.uniform(low, high))


def combine_parents(population: List) -> List[Tuple]:
    """Return all possible combinations for crossover given a list of (selected) parents."""
    return list(combinations(population, 2))


# Reax Parsing Helpers
def get_param(key: List[int], ffield: Dict[int, List]) -> float:
    """Using a key in the params key list, find the corresponding value (parameter) in the ffield dictionary.

    Parameters
    ----------
    key: List[int]
        list containing 3 columns: [section # | line/row # | parameter #].
    ffield: Dict[int, List]
        ffield[section number] = [parameters corresponding to section].

    Returns
    -------
    value: float
        Value in FFIELD file at location of key.
    """
    section = key[0]
    row = key[1] - 1  # Indices start at 0 in Python

    return ffield[section][row][key[2] - 1]


def set_param(key: List[int], ffield: Dict[int, List], param: float):
    """Using a key in the params key list, set the corresponding parameter in the given Child's dictionary.

    Parameters
    ----------
    key: List[int]
        list containing 3 columns: [section # | line/row # | parameter #]
    ffield: Dict[int, List]
        ffield[section number] = [parameters corresponding to section].
    param: float
        Value of the parameter to store in ffield dict at the specified key
    """
    section = key[0]
    row = key[1] - 1  # Indices start at 0 in Python
    ffield[section][row][key[2] - 1] = param


# Remove duplicates from Params data structures
def unique_params(param_keys: List, param_increments: List, param_bounds: List) -> Tuple[List, List, List]:
    """Removes duplicates from param_keys list, and removes corresponding values in
    param_increments and param_bounds. Returns new param_keys, param_increments, and
    param_bounds lists.

    Returns
    -------
    final_param_keys: List[List[int]]
        param_keys list without duplicates (order preserved)
    final_param_increments: List[int]
        param_increments list with items corresponding to unique param_keys (order preserved)
    final_param_bounds: List[List[float, float]]
        param_bounds list with items corresponding to unique param_keys (order preserved)
    """
    final_param_keys = []
    final_param_increments = []
    final_param_bounds = []

    for i in range(len(param_keys)):
        if param_keys[i] not in final_param_keys:
            final_param_keys.append(param_keys[i])
            final_param_increments.append(param_increments[i])
            if len(param_bounds[i]) == 2 and param_bounds[i][1] < param_bounds[i][0]:
                # Upper bound is smaller than lower bound
                param_bounds[i][0], param_bounds[i][1] = param_bounds[i][1], param_bounds[i][0]
            final_param_bounds.append(param_bounds[i])

    return final_param_keys, final_param_increments, final_param_bounds


def uniques(arr: List, order_matters: bool = True) -> List:
    """Removes duplicate values in a list and returns a new list.
    If order of list does not matter, use a more efficient method depending
    on sets. If order matters, use less efficient but simple method to preserve
    original list ordering.

    Parameters
    ----------
    arr: List
        List to remove duplicates from.
    order_matters: bool, default = True
        Whether or not the original ordering of the list must be preserved.

    Returns
    -------
    final: List
        List without duplicates.
    """
    if not order_matters:
        # CASE 1: ORDER DOES NOT MATTER
        # Need to convert the inner lists to tuples so they are hashable
        unique_set = set(map(tuple, arr))
        final = [list(item) for item in unique_set]  # Converting set of tuples back to list of lists
    else:
        # CASE 2: ORDER MATTERS
        # More inefficient than case 1, but easy method to keep ordering of original list
        final = []
        for item in arr:
            if item not in final:
                final.append(item)

    return final


# Miscellaneous helpers
def pop_random(arr: List):
    """Remove and return randomly picked element from list."""
    idx = random.randrange(0, len(arr))
    return arr.pop(idx)
