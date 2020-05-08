#!/usr/bin/env python

"""Module with densely-connected Feed-Forward Neural Network with one hidden layer.
Uses Keras with TensorFlow backend to build the neural network.
"""

# Standard library
from typing import List

# 3rd party packages
import tensorflow as tf

# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.domain.cost.strategy import IErrorStrategy
from parametrization_clean.domain.neural_network.extract_data import (individuals_to_features_df,
                                                                      individuals_to_features_and_outputs_df,
                                                                      extract_features_and_outputs_from_combined_df)
from parametrization_clean.domain.neural_network.transform_data import (train_test_split, normalize_features,
                                                                        get_columns_to_remove,
                                                                        remove_problematic_columns)


class FeedForwardNet:

    def __init__(self, population: List[Individual], verbosity: int = 2, train_fraction: float = 0.80,
                 num_epochs: int = 20000):
        self.population = population
        self.num_input_nodes = len(population[0].params)
        self.num_output_nodes = len(population[0].reax_energies)

        self.train_fraction = train_fraction
        self.verbosity = verbosity
        self.num_epochs = num_epochs

        x_y_df = individuals_to_features_and_outputs_df(population)
        train_df, test_df = train_test_split(x_y_df, train_fraction)
        train_x, self.train_y = extract_features_and_outputs_from_combined_df(train_df, self.num_input_nodes)
        test_x, self.test_y = extract_features_and_outputs_from_combined_df(test_df, self.num_input_nodes)
        self.columns_to_remove = get_columns_to_remove(train_x)
        self.train_x = remove_problematic_columns(train_x, self.columns_to_remove)
        self.test_x = remove_problematic_columns(test_x, self.columns_to_remove)

        self.train_stats = self.train_x.describe().transpose()
        self.normalized_train_x = normalize_features(self.train_x, self.train_stats)
        self.normalized_test_x = normalize_features(self.test_x, self.train_stats)

    def execute(self):
        model = self.build()
        history = self.fit(model)
        return model, history

    def build(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(int(self.num_input_nodes), input_shape=[self.num_input_nodes], activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),  # Dropout on the visible (input) layer - rate = frac. inputs to drop
                tf.keras.layers.Dense(self.num_output_nodes, activation='linear',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.1)
                                      )
            ]
        )
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=[r_square, 'mse'])
        return model

    def fit(self, model):
        """Perform fitting for ANN model to [x, y] data and return its history."""
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        history = model.fit(self.normalized_train_x, self.train_y, epochs=self.num_epochs, batch_size=32,
                            validation_split=0.2, verbose=self.verbosity,
                            callbacks=[
                                early_stop,
                            ]
                            )
        return history

    def predict_outputs(self, model, population: List[Individual]):
        """Given an ANN model and a population, predict the associated outputs with that population."""
        x_df = individuals_to_features_df(population)
        cleaned_x_df = remove_problematic_columns(x_df, self.columns_to_remove)
        normalized_x_df = normalize_features(cleaned_x_df, self.train_stats)
        y_predicted = model.predict(normalized_x_df)
        return y_predicted

    @staticmethod
    def compute_costs(y_predicted, root_individual: RootIndividual, error_strategy: IErrorStrategy, **kwargs) \
            -> List[float]:
        dft_energies = root_individual.dft_energies
        weights = root_individual.weights
        costs = [sum(error_strategy.error(reax_pred, dft_energy, weight, **kwargs)
                     for reax_pred, dft_energy, weight in zip(y_pred_row, dft_energies, weights))
                 for y_pred_row in y_predicted]
        return costs


def r_square(y_true, y_pred):
    """Coefficient of determination (R^2) for regression - only for Keras tensors.
    To be used in metrics args.
    """
    ss_residual = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    ss_total = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - ss_residual / (ss_total + tf.keras.backend.epsilon())
