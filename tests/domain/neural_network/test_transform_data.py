# Standard library

# 3rd party packages
import pytest
import numpy as np
import pandas as pd

# Local source
from parametrization_clean.domain.neural_network.extract_data import (individuals_to_features_df,
                                                                      individuals_to_features_and_outputs_df)
from parametrization_clean.domain.neural_network.transform_data import (train_test_split, normalize_features,
                                                                        denormalize_features,
                                                                        get_columns_to_remove,
                                                                        remove_problematic_columns)


@pytest.fixture()
@pytest.mark.usefixtures('get_individuals')
def x_df(get_individuals):
    return individuals_to_features_df(get_individuals)


@pytest.fixture()
@pytest.mark.usefixtures('get_individuals')
def x_y_df(get_individuals):
    return individuals_to_features_and_outputs_df(get_individuals)


@pytest.fixture()
def x_train_stats(x_df):
    return x_df.describe().transpose()


def test_train_test_split(x_y_df):
    train_fraction = 0.75
    train_df, test_df = train_test_split(x_y_df, train_fraction)
    assert train_df.shape[0] == 3
    assert test_df.shape[0] == 1
    assert train_df.shape[1] == 9
    assert test_df.shape[1] == 9


def test_normalize_features(x_df, x_train_stats):
    normalized_x_df = normalize_features(x_df, x_train_stats)
    normalized_x_nd_array = normalized_x_df.to_numpy()
    expected_normalized_array = np.array([[0.412005595, -0.488850326, -0.349418121, -1.178611022, 0.968101956],
                                          [-1.142732499, -1.063968358, -1.214643944, 0.483037304, -0.033382826],
                                          [1.15827988, 0.316314917, 1.081532279, -0.405751336, 0.433976739],
                                          [-0.427552976, 1.236503767, 0.482529786, 1.101325054, -1.368695868]
                                          ])
    for param, expected_param in np.nditer([normalized_x_nd_array, expected_normalized_array]):
        assert param == pytest.approx(expected_param)


def test_denormalize_features(x_df, x_train_stats):
    normalized_x_df = normalize_features(x_df, x_train_stats)
    denormalized_x_df = denormalize_features(normalized_x_df, x_train_stats)
    denormalized_x_nd_array = denormalized_x_df.to_numpy()
    expected_denormalized_array = np.array([[1.0, 0.1, -0.5, 4.3, 2.4],
                                            [0.5, 0.05, -0.76, 8.6, 2.1],
                                            [1.24, 0.17, -0.07, 6.3, 2.24],
                                            [0.73, 0.25, -0.25, 10.2, 1.7]
                                            ])
    for param, expected_param in np.nditer([denormalized_x_nd_array, expected_denormalized_array]):
        assert param == pytest.approx(expected_param)


def test_get_columns_to_remove():
    x_df = pd.DataFrame([[1.0, 0.1, -0.5, 4.3, 2.4],
                         [0.5, 0.1, -0.76, 8.6, 2.1],
                         [1.24, 0.1, -0.07, 6.3, 2.24],
                         [0.73, 0.1, -0.25, 10.2, 1.7]
                         ])
    cols_to_remove = get_columns_to_remove(x_df)
    assert cols_to_remove == [1]


def test_remove_problematic_columns():
    x_df = pd.DataFrame([[1.0, 0.1, -0.5, 4.3, 2.4],
                         [0.5, 0.1, -0.76, 8.6, 2.1],
                         [1.24, 0.1, -0.07, 6.3, 2.24],
                         [0.73, 0.1, -0.25, 10.2, 1.7]
                         ])
    columns_to_remove = get_columns_to_remove(x_df)
    new_x_df = remove_problematic_columns(x_df, columns_to_remove)
    new_x_nd_array = new_x_df.to_numpy()
    expected_new_array = np.array([[1.0, -0.5, 4.3, 2.4],
                                   [0.5, -0.76, 8.6, 2.1],
                                   [1.24, -0.07, 6.3, 2.24],
                                   [0.73, -0.25, 10.2, 1.7]
                                   ])
    assert new_x_nd_array.shape == expected_new_array.shape
    for param, expected_param in np.nditer([new_x_nd_array, expected_new_array]):
        assert param == expected_param
