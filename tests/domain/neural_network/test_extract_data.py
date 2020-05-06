
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.neural_network.extract_data import (individuals_to_features_df,
                                                                      individuals_to_features_and_outputs_df,
                                                                      extract_features_and_outputs_from_combined_df)


@pytest.mark.usefixtures('get_individuals')
def test_individuals_to_features_df(get_individuals):
    features_df = individuals_to_features_df(get_individuals)
    params_nd_array = features_df.to_numpy()
    assert list(params_nd_array[0, :]) == [1.0, 0.1, -0.5, 4.3, 2.4]
    assert list(params_nd_array[1, :]) == [0.5, 0.05, -0.76, 8.6, 2.1]
    assert list(params_nd_array[2, :]) == [1.24, 0.17, -0.07, 6.3, 2.24]
    assert list(params_nd_array[3, :]) == [0.73, 0.25, -0.25, 10.2, 1.7]


@pytest.mark.usefixtures('get_individuals')
def test_individuals_to_features_and_outputs_df(get_individuals):
    features_and_outputs_df = individuals_to_features_and_outputs_df(get_individuals)
    features_and_outputs_nd_array = features_and_outputs_df.to_numpy()
    assert list(features_and_outputs_nd_array[0, :]) == [1.0, 0.1, -0.5, 4.3, 2.4, 43.2, 10.6, -174.2, 1008.4]
    assert list(features_and_outputs_nd_array[1, :]) == [0.5, 0.05, -0.76, 8.6, 2.1, 56.9, 11.2, -164.1, 994.8]
    assert list(features_and_outputs_nd_array[2, :]) == [1.24, 0.17, -0.07, 6.3, 2.24, 102.4, 5.5, -155.5, 1008.12]
    assert list(features_and_outputs_nd_array[3, :]) == [0.73, 0.25, -0.25, 10.2, 1.7, 40.2, 11.2, -104.1, 1018.32]


@pytest.mark.usefixtures('get_individuals')
def test_extract_features_and_outputs_from_combined_df(get_individuals):
    features_and_outputs_df = individuals_to_features_and_outputs_df(get_individuals)
    num_features = len(get_individuals[0].params)
    x_df, y_df = extract_features_and_outputs_from_combined_df(features_and_outputs_df, num_features)
    params_nd_array = x_df.to_numpy()
    reax_energies_nd_array = y_df.to_numpy()
    assert list(params_nd_array[0, :]) == [1.0, 0.1, -0.5, 4.3, 2.4]
    assert list(params_nd_array[1, :]) == [0.5, 0.05, -0.76, 8.6, 2.1]
    assert list(params_nd_array[2, :]) == [1.24, 0.17, -0.07, 6.3, 2.24]
    assert list(params_nd_array[3, :]) == [0.73, 0.25, -0.25, 10.2, 1.7]
    assert list(reax_energies_nd_array[0, :]) == [43.2, 10.6, -174.2, 1008.4]
    assert list(reax_energies_nd_array[1, :]) == [56.9, 11.2, -164.1, 994.8]
    assert list(reax_energies_nd_array[2, :]) == [102.4, 5.5, -155.5, 1008.12]
    assert list(reax_energies_nd_array[3, :]) == [40.2, 11.2, -104.1, 1018.32]
