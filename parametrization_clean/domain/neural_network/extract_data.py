#!/usr/bin/env python

"""Module to extract Individual's data to a convenient form for neural network training."""

# Standard library
from typing import List, Tuple

# 3rd party packages
import pandas as pd

# Local source
from parametrization_clean.domain.individual import Individual


def individuals_to_features_df(population: List[Individual]) -> pd.DataFrame:
    return pd.DataFrame(data=[individual.params for individual in population])


def individuals_to_features_and_outputs_df(population: List[Individual]) -> pd.DataFrame:
    params_and_reax_predictions = [individual.params + individual.reax_energies for individual in population]
    return pd.DataFrame(data=params_and_reax_predictions)


def extract_features_and_outputs_from_combined_df(x_y_df: pd.DataFrame, num_features: int) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_df = x_y_df.iloc[:, 0:num_features]
    y_df = x_y_df.iloc[:, num_features:]
    return x_df, y_df
