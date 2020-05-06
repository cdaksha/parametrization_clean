#!/usr/bin/env python

"""Module to transform data into training and test sets."""

# Standard library
from typing import Tuple

# 3rd party packages
import pandas as pd

# Local source


def train_test_split(x_y_df: pd.DataFrame, train_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets."""
    train_df = x_y_df.sample(frac=train_fraction)
    test_df = x_y_df.drop(train_df.index)
    return train_df, test_df


def normalize_features(x_df: pd.DataFrame, x_train_stats: pd.DataFrame) -> pd.DataFrame:
    """Normalize the dataset according to training data statistics (obtained from pandas)."""
    return (x_df - x_train_stats['mean']) / x_train_stats['std']


def denormalize_features(x_df: pd.DataFrame, x_train_stats: pd.DataFrame) -> pd.DataFrame:
    """De-normalize the dataset according to the training data statistics (obtained from pandas)."""
    return (x_df * x_train_stats['std']) + x_train_stats['mean']


def get_columns_to_remove(x_df: pd.DataFrame, tolerance: float = 1e-4):
    """Preprocessing step for training ANN. Columns to remove from features dataframe are those with standard
    deviations that are too small and will result in normalization issues.
    """
    return x_df.std()[abs(x_df.std()) <= tolerance].index


def remove_problematic_columns(x_df, columns_to_remove):
    """Return new df with problematic columns removed."""
    return x_df.drop(columns_to_remove, axis=1)
