#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorFlow 2.0 playing grounds.

__author__ = "Chad Daksha"
"""

# Standard library


# 3rd party packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local source
import core.neural_net.ann as ann
from core.neural_net.visualize import plot_history
from core.settings.config import settings as s
from core.genetic_algorithm.helpers import reax_error

# Global constant
FMT = '-' * 50  # for separating chunks of print statements

# Reconfiguring constants in settings file
s.ANN.trainSplit = 0.80
s.ANN.validationSplit = 0.20
s.ANN.epochs = 2000
num_generations_to_read = 10  # APPROXIMATE

# Retrieving raw data
data_path = "C:\\Users\\chadd\\Desktop\\000-Parametrization-Results\\ZnO-parametrization\\" \
            "00-ANN-training-data\\base_case_3"
X = pd.read_csv(data_path + '\\x_data.csv').iloc[:, 1:]
y = pd.read_csv(data_path + '\\y_data.csv').iloc[:, 1:]
weights = pd.read_csv(data_path + '\\weights.csv').values.tolist()
true_vals = pd.read_csv(data_path + '\\DFT_values.csv').values.tolist()
num_cases_to_consider = num_generations_to_read * 50  # popSize = 50
X = X.iloc[0:num_cases_to_consider, :]
y = y.iloc[0:num_cases_to_consider, :]

# TESTING ROOT PARAMS
s.path.root = 'C:\\Users\\chadd\\Desktop\\000-Parametrization-Results\\00-training-sets\\' \
              'test-sets\\ReaxFF_ZnO_Raymand_with_Sglass_control_all_bounds'
from core.genetic_algorithm.population import Population
pop_controller = Population()
print(pop_controller._root.params)

# Data preparation
# For objective function calculation
total_errors = [sum(reax_error(reax_pred=x, true_val=y, weight=1)
                    for x, y in zip(vals_row, true_vals))
                for vals_row in np.array(y)]

# Establishing baseline - average total error
baseline_prediction = np.mean(total_errors)
baseline_errors = [abs(baseline_prediction - total_error) for total_error in total_errors]
print('Average baseline error: {:.3f}'.format(np.mean(baseline_errors)))

# Fitting ANN
# TODO: fix
input_size = X.shape[1]
output_size = y.shape[1]
X_y_df = pd.concat([X, y], axis=1, sort=False)

train_ds, test_ds = ann.train_test_split(X_y_df, s.ANN.trainSplit)
train_x = train_ds.iloc[:, 0:input_size]
train_y = train_ds.iloc[:, input_size:]
test_x = test_ds.iloc[:, 0:input_size]
test_y = test_ds.iloc[:, input_size:]

# Drop the columns where all elements are equal
cols_to_drop = X.std()[abs(X.std()) <= 1e-4].index
test_x = test_x.drop(cols_to_drop, axis=1)
train_x = train_x.drop(cols_to_drop, axis=1)
train_stats = train_x.describe().transpose()

model = ann.build(input_size, output_size)
normed_train_x = ann.normalize(train_x, train_stats)
normed_test_x = ann.normalize(test_x, train_stats)

history = ann.fit(normed_train_x, train_y, model, s.ANN.epochs)
loss, r_squared, mse = model.evaluate(normed_test_x, test_y, verbose=2)
print("Testing set R-squared: {:5.2f}".format(r_squared))

model.summary()
hist = pd.DataFrame(history.history)
final_results = hist.tail(1)

parents_predicted_output = model.predict(ann.normalize(X, train_stats))

print(y)
print(pd.DataFrame(parents_predicted_output))

# Actual vs. Predicted Total ReaxFF Error
predicted_errors = [[reax_error(reax_pred=x, true_val=y, weight=1)
                     for x, y, z in zip(vals_row, true_vals, weights)]
                    for vals_row in parents_predicted_output]
predicted_total_errors = [sum(error_list) for error_list in predicted_errors]
print(total_errors)
print(predicted_total_errors)

# Visualization
plt.close('all')
plt.style.use('ggplot')
plot_history(history)

absolute_difference = [actual_error - predicted_error
                       for actual_error, predicted_error in zip(total_errors, predicted_total_errors)]
print(absolute_difference)
plt.hist(absolute_difference)
plt.xlabel('Difference between Actual and Predicted ReaxFF Objective Function')
plt.ylabel('Frequency')
plt.show()


# First column
first_reax_val = y.iloc[:, 0]
first_predicted_val = parents_predicted_output[:, 0]
plt.scatter(first_reax_val, first_predicted_val)
plt.xlabel('True ReaxFF Value')
plt.ylabel('Predicted ReaxFF Value')
plt.xlim([min(first_reax_val), max(first_reax_val)])
plt.ylim([min(first_reax_val), max(first_reax_val)])
plt.show()


# Second column
reax_val = y.iloc[:, 1]
predicted_val = parents_predicted_output[:, 1]
plt.scatter(reax_val, predicted_val)
plt.xlabel('True ReaxFF Value')
plt.ylabel('Predicted ReaxFF Value')
plt.xlim([min(reax_val), max(reax_val)])
plt.ylim([min(reax_val), max(reax_val)])
plt.show()
