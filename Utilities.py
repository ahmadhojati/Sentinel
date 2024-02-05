#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import copy
import matplotlib.pyplot as plt

def round_coordinates(coord):
    """
    Round latitude and longitude coordinates to 5 decimals (~ 1 m accuracy).

    Parameters:
    - coord: Tuple containing latitude and longitude coordinates.

    Returns:
    - Rounded coordinates.
    """
    return (round(coord[0], 5), round(coord[1], 5))


def determine_water_year_start(year):
    """
    Determine the water year start date based on the given year.

    Parameters:
    - year: Year for which to determine the water year start date.

    Returns:
    - Water year start date.
    """
    # You can customize this logic to determine the water year start based on your needs.
    # For this example, we'll assume a water year starts on October 1st for all years.
    return pd.to_datetime(f'{year}-10-01')


def calculate_r_squared_inv(y_true, y_pred):
    """
    Calculate the inverse of R-squared (coefficient of determination) for a simple linear regression.

    Parameters:
    - y_true: List or array of true target values.
    - y_pred: List or array of predicted target values.

    Returns:
    - Inverse of R-squared value.
    """
    mean_y_true = sum(y_true) / len(y_true)
    tss = sum((y - mean_y_true) ** 2 for y in y_true)
    rss = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    r_squared = 1 - (rss / tss)
    return (1 - r_squared)


def det_coeff(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) using Keras backend operations.

    Parameters:
    - y_true: List or array of true target values.
    - y_pred: List or array of predicted target values.

    Returns:
    - R-squared value.
    """
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))


def feature_importance_(X, y, loaded_model):
    """
    Calculate feature importance using mean squared error (MSE) and R-squared.

    Parameters:
    - X: Input features.
    - y: Target values.
    - loaded_model: Trained machine learning model.

    Returns:
    - Array containing MSE and R-squared values for each feature.
    """
    np.random.seed(42)
    permuted_train_test = copy.deepcopy(X)
    MSE_R_2 = np.empty((permuted_train_test.shape[1], 2))

    for variable in range(permuted_train_test.shape[1]):
        permuted_train_test = copy.deepcopy(X)

        np.apply_along_axis(np.random.shuffle, axis=-1, arr=permuted_train_test[:, variable])

        MSE_R_2[variable] = loaded_model.evaluate(permuted_train_test, y, batch_size=len(permuted_train_test), verbose=0)

    return MSE_R_2


def plot_feature_importance(MSE_R_2, label, arg, save=False):
    """
    Create a bar plot to visualize feature importance.

    Parameters:
    - MSE_R_2: Array containing MSE and R-squared values for each feature.
    - label: Feature labels.
    - arg: Additional arguments.
    - save: Boolean indicating whether to save the plot.

    Returns:
    - None (plots the feature importance).
    """
    X = np.arange(MSE_R_2.shape[0])
    fig = plt.figure(figsize=(20, 10))

    MSE_R_2_normal = min_max_normalize(np.sqrt(MSE_R_2[:, 0]))

    plt.bar(X + 0.00, MSE_R_2_normal, color='b', width=0.25, label='Normalized RMSE')
    plt.bar(X + 0.25, MSE_R_2[:, 1], color='g', width=0.25, label='R-squared')
    objects = label
    plt.xticks(X, objects, rotation=15, size=18)
    plt.legend(prop={'size': 24})

    if save:
        plt.savefig('{}.png'.format(arg), dpi=300)


def min_max_normalize(data):
    """
    Perform min-max normalization on a given dataset.

    Parameters:
    - data: Input data to be normalized.

    Returns:
    - Normalized data.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

