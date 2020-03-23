#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Random Forest regression playing grounds.

__author__ = "Chad Daksha"
"""

# Standard library


# 3rd party packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Local source
import core.utils.population_io as pop_io
from core.settings.config import settings as s

# Global constant
FMT = '-' * 50  # for separating chunks of print statements

# Reconfiguring constants in settings file
s.path.population = 'C:\\Users\\chadd\\Desktop\\000-Parametrization-Results\\ZnO-parametrization\\' \
                    'test-generational-data\\'
s.path.root = s.path.population + 'generation-150\\child-0'

# Retrieving raw data
pop_reader = pop_io.PopulationIO(generation_num=161)  # Reads data for (generation_num - 1)
parents = pop_reader.read_previous_n_populations(num=10)

# Data preparation
total_errors = np.array([parent.cost for parent in parents])
X = pd.DataFrame([parent.params for parent in parents])
feature_list = list(X.columns)
# Converting to numpy array
X = np.array(X)
train_X, test_X, train_Y, test_Y = train_test_split(X, total_errors, test_size=0.1, random_state=0)
# train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)

print("{fmt}Input Data Size{fmt}".format(fmt=FMT))
print('Training Features Shape:', train_X.shape)
print('Training Labels Shape:', train_Y.shape)
# print('Validation Features Shape:', val_X.shape)
# print('Validation Labels Shape:', val_Y.shape)
print('Testing Features Shape:', test_X.shape)
print('Testing Labels Shape:', test_Y.shape)
print()

# Establishing baseline - average total error
baseline_prediction = np.mean(total_errors)
baseline_errors = [abs(baseline_prediction - total_error) for total_error in total_errors]
print("{fmt}Random Forest Regressor Accuracy{fmt}".format(fmt=FMT))
print('Average baseline error: {:.3f}'.format(np.mean(baseline_errors)))

# ---------------------
# Train random forest regressor with 1000 decision trees
# rf = RandomForestRegressor(n_estimators=500, random_state=0)
# rf.fit(train_X, train_Y)
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=2000, num=200)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf.fit(train_X, train_Y)
print("BEST RANDOM FOREST PARAMETERS: ")
print(rf.best_params_)
# ----------------------


def evaluate(model, train_features, train_labels, test_features, test_labels):
    pred = model.predict(test_features)
    print('Model Performance')
    print('Training R-squared = {:.2f}'.format(model.score(train_features, train_labels)))
    print('Testing R-squared = {:.2f}.'.format(model.score(test_features, test_labels)))
    return pred


# BASE MODEL
base_rf = RandomForestRegressor(n_estimators=500, random_state=0)
base_rf.fit(train_X, train_Y)
evaluate(base_rf, train_X, train_Y, test_X, test_Y)
evaluate(rf, train_X, train_Y, test_X, test_Y)

# Use model to predict for test set
predictions = rf.predict(test_X)
absolute_errors = abs(predictions - test_Y)
# squared_errors = (predictions - test_Y) ** 2
print("Mean Absolute Error: {:.3f}".format(np.mean(absolute_errors)))
# print("Training R-squared: {:.2f}".format(rf.score(train_X, train_Y)))
# print("Validation R-squared: {:.2f}".format(rf.score(val_X, val_Y)))
# print("Testing R-squared: {:.2f}".format(rf.score(test_X, test_Y)))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (absolute_errors / test_Y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print("Testing Accuracy (based on MAPE): {:.2f}%.".format(accuracy))
print()

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2))
                       for feature, importance in enumerate(importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# Print out the feature and importances
# print("{fmt}Variable Importances{fmt}".format(fmt=FMT))
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Visualizations
plt.close('all')
# Variable importances
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation='vertical')
# Tick labels for x axis
# plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()

# Actual vs. prediction
# Graph labels
train_predictions = rf.predict(train_X)
# plt.scatter(train_Y, train_predictions)
plt.scatter(test_Y, predictions)
# plt.plot([train_Y.min(), train_Y.max()], [train_Y.min(), train_Y.max()], 'k--', lw=4)
plt.plot([test_Y.min(), test_Y.max()], [test_Y.min(), test_Y.max()], 'k--', lw=2)
plt.xlabel('Actual Total Error')
plt.ylabel('Predicted Total Error')
plt.show()

[print("actual: {:.2f}\tpredicted: {:.2f}".format(actual, predicted)) for actual, predicted in zip(test_Y, predictions)]
