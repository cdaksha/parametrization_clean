#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nested Genetic Algorithm implementation.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import List
import os

# 3rd party packages
import pandas as pd

# Local source
from core.genetic_algorithm import individual as child, population as pop, helpers as util
from core.neural_net.ann import normalize
from core.settings.config import settings as s


def nested_ga(num_iters: int, master_parents: List[child.Individual], model, train_stats,
              pop_controller: pop.Population) -> List[child.Individual]:
    """Nested Genetic Algorithm loop that uses predictions from `model` (ANN) to determine lowest cost."""

    # For error calculation
    true_vals = master_parents[0].true_vals
    weights = master_parents[0].weights

    master_parents_x = pd.DataFrame(data=[parent.params for parent in master_parents])

    # Remove columns where all elements are equal to prevent std = 0 (leading to NaN from normalization)
    cols_to_drop = master_parents_x.std()[abs(master_parents_x.std()) <= 1e-4].index

    master_parents_x = master_parents_x.drop(cols_to_drop, axis=1)
    master_parents_y = pd.DataFrame(data=[parent.reax_predictions for parent in master_parents])

    master_predicted_output = pd.DataFrame(model.predict(normalize(master_parents_x, train_stats)))
    error = master_parents_y.values - master_predicted_output.values
    print("{fmt}MASTER ReaxFF VALUES{fmt}".format(fmt='-'*50))
    print(master_parents_y)
    print("{fmt}PREDICTED ReaxFF VALUES{fmt}".format(fmt='-'*50))
    print(master_predicted_output)
    print("{fmt}DIFFERENCE BETWEEN TRUE AND PREDICTED{fmt}".format(fmt='-' * 50))
    print(error)

    # First population
    current_parents = pop_controller.propagate(master_parents)
    # Preserve best two from master GA
    best_master_parents = current_parents[0:2]

    # Propagation
    for i in range(num_iters):
        if i > 0:
            current_parents = pop_controller.propagate(current_parents)

        # Data frame conversion required for Keras formatting
        current_parents_df = pd.DataFrame(data=[parent.params for parent in current_parents])
        # Remove columns where all elements are equal to prevent std = 0 (leading to NaN from normalization)
        current_parents_df = current_parents_df.drop(cols_to_drop, axis=1)

        # Predicted reaxFF values from ANN
        # TESTING using total error instead of reax values as Y
        predicted_output = model.predict(normalize(current_parents_df, train_stats))

        # TODO: change this to predict the composite fitness index?
        # TODO: implement comparison of ANN total objective function output vs. true values
        # List[List] with all errors for each child in generation
        errors = [[util.reax_error(reax_pred=x, true_val=y, weight=z)
                   for x, y, z in zip(vals_row, true_vals, weights)]
                  for vals_row in predicted_output]

        for current_child, errors_row in zip(current_parents, errors):
            current_child.reax_predictions = predicted_output
            current_child.errors = errors_row
            current_child.cost = sum(current_child.errors)

        all_errors = [current_child.cost for current_child in current_parents]
        print("Nested GA Generation {}: Predicted Minimum Error = {}".format(i, min(all_errors)))

    final_parents = best_master_parents + current_parents[0:(s.GA.populationSize - 2)]
    return final_parents
