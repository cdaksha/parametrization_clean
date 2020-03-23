#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorFlow 2.0 playing grounds.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import Tuple, List
import os

# 3rd party packages
import tensorflow as tf
import pandas as pd

# Local source
from core.neural_net.helpers import to_df
from core.genetic_algorithm import individual as c
from core.settings.config import settings as s

POPULATION_PATH = s.path.population
ANN_FILE_NAME = s.path.ANNFile


# TODO: Still need to finish refactoring this section
# TODO: move train_test_split, normalize, denormalize, nested_GA to a Facade?
def train_test_split(df: pd.DataFrame, train_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets."""
    train_df = df.sample(frac=train_frac)
    test_df = df.drop(train_df.index)

    return train_df, test_df


# Standardization to prevent overfitting
def normalize(df: pd.DataFrame, train_stats: pd.DataFrame) -> pd.DataFrame:
    """Normalize the dataset according to training data statistics."""
    return (df - train_stats['mean']) / train_stats['std']


def denormalize(df: pd.DataFrame, train_stats: pd.DataFrame) -> pd.DataFrame:
    """De-normalize the dataset according to the training data statistics."""
    return (df * train_stats['std']) + train_stats['mean']


def r_square(y_true, y_pred):
    """Coefficient of determination (R^2) for regression - only for Keras tensors.
    To be used in metrics args.
    """
    ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    ss_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


# Model Creation
def build(num_input, num_output):
    """Densely connected ANN with one hidden layer."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(int(num_input), input_shape=[num_input], activation='relu'),
            tf.keras.layers.Dropout(rate=0.2),  # Dropout on the visible (input) layer - rate = frac. inputs to drop
            tf.keras.layers.Dense(num_output, activation='linear',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.1)
                                  )
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=[r_square, 'mse'])

    return model


def fit(train_x, train_y, model, epochs):
    """Perform fitting for ANN model to [x, y] and return its history."""
    # Callbacks
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    # Save model makes model a lot more time consuming for some reason?
    # ann_path = os.path.join(s.population_path, s.ANN_PATH)
    # save_model = tf.keras.callbacks.ModelCheckpoint(ann_path, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(train_x, train_y, epochs=epochs, batch_size=32,
                        validation_split=0.2, verbose=2,
                        callbacks=[
                            early_stop,
                            # save_model,
                        ]
                        )

    return history


def run(parents: List[c.Individual], epochs: int, generation_num: int):
    """Combine `ann` module functionalities to fit ANN to `parents` of a genetic algorithm generation.

    Loads the model from `generation_num - 1` if `generation_num - 1` > 1.
    """
    input_size = len(parents[0].params)
    output_size = len(parents[0].reax_predictions)

    # if (generation_num - 1) == 1:
    #    model = build(num_input=input_size, num_output=output_size)
    # else:
    #    model = load(generation_num - 1)
    # Never loading the model - always rebuilding

    x_y_df = to_df(parents)
    train_ds, test_ds = train_test_split(x_y_df, s.ANN.trainSplit)

    train_x = train_ds.iloc[:, 0:input_size]
    train_y = train_ds.iloc[:, input_size:]
    test_x = test_ds.iloc[:, 0:input_size]
    test_y = test_ds.iloc[:, input_size:]

    # Drop the columns where all elements are equal
    test_x = test_x.drop(train_x.std()[abs(train_x.std()) <= 1e-4].index, axis=1)
    train_x = train_x.drop(train_x.std()[abs(train_x.std()) <= 1e-4].index, axis=1)

    train_stats = train_x.describe().transpose()

    model = build(train_x.shape[1], output_size)

    normed_train_x = normalize(train_x, train_stats)
    normed_test_x = normalize(test_x, train_stats)

    history = fit(normed_train_x, train_y, model, epochs)

    loss, r_squared, mse = model.evaluate(normed_test_x, test_y, verbose=2)

    print("Testing set R-squared: {:5.2f}".format(r_squared))

    return model, history, train_stats


# Model I/O
def load(generation_num: int):
    """Load ANN saved in population_path/generation-`generation_num`/ANN_PATH."""
    # When loading the model, custom metrics must be stored.
    dependencies = {
        'r_square': r_square,
    }
    ann_path = os.path.join(POPULATION_PATH, 'generation-' + str(generation_num), ANN_FILE_NAME)
    return tf.keras.models.load_model(ann_path, custom_objects=dependencies)


def save(model, generation_num: int):
    """Save ANN model to population_path/generation-`generation_num`/ANN_PATH."""
    ann_path = os.path.join(POPULATION_PATH, 'generation-' + str(generation_num), ANN_FILE_NAME)
    model.save(ann_path)
