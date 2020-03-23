#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization functions for neural network implementation.

__author__ = "Chad Daksha"
"""

# Standard library

# 3rd party packages
import pandas as pd
import matplotlib.pyplot as plt

# Local source


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['r_square'],
             label='Train $R^2$')
    plt.plot(hist['epoch'], hist['val_r_square'],
             label='Val $R^2$')
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.legend()
    plt.show()
