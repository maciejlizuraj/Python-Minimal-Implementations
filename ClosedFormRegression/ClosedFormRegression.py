"""
This script performs linear regression on a given dataset using Closed-Form Solution
(Normal Equation)

Should be run with parameters: <input file> <degree of polynomial>

The script reads data from a CSV file, prepares it for the chosen algorithm,
performs the regression, and saves the results to a CSV file and a plot image.
"""
import sys
from typing import Tuple
import numpy.typing as npt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_FILE_NAME = "equation.csv"
PLOT_FILE_NAME = "plot.png"


def read_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads data from a CSV file.

    Args:
        filename (str): Path to the CSV data file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the features (x) and target variable (y)
        as NumPy arrays.
    """
    df = pd.read_csv(filename)
    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return x, y


def prepare_data(x: np.ndarray, y: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data for the closed-form solution algorithm.

    Args:
        x (np.ndarray): The feature array.
        y (np.ndarray): The target variable array.
        degree (int): The degree of the polynomial for which regression is performed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the prepared X and y arrays.
    """
    number_of_samples = x.shape[0]
    X = np.c_[np.ones((number_of_samples, 1)), x]
    for i in range(2, degree + 1):
        X = np.c_[X, x ** i]
    y = y.reshape(number_of_samples, 1)
    return X, y


def calculate_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the theta parameters using the closed-form solution (Normal Equation).

    Args:
        X (np.ndarray): The features matrix.
        y (np.ndarray): The target variable array.

    Returns:
        np.ndarray: The theta vector containing the model parameters.
    """
    tmp = X.T.dot(X)
    tmp = np.linalg.inv(tmp)
    tmp = tmp.dot(X.T)
    theta_hat = tmp.dot(y)
    return theta_hat


def calculate_hypothesis(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Calculates the predicted target values (y-hat) using the linear regression model.

    Args:
        X (np.ndarray): The features matrix.
        theta (np.ndarray): The theta vector containing the model parameters.

    Returns:
        np.ndarray: The predicted y-hat values as a NumPy array.
    """
    y = theta.T.dot(X.T)
    return y


def temp_func(filename: str, degree: int) -> None:
    """
    The main function that performs the linear regression.

    Args:
        filename (str): Path to the CSV data file.
        degree (int): The degree of the polynomial for the closed-form solution.
    """
    x, y = read_data(filename)
    y_pred: npt.NDArray[np.float64] = np.ndarray([0])
    X, y = prepare_data(x, y, degree)
    theta_hat = calculate_normal_equation(X, y)
    y_pred = calculate_hypothesis(X, theta_hat).flatten()

    plt.scatter(x[:, 0], y, c='green')
    plt.plot(x[:, 0], y_pred, c='red')

    df = pd.DataFrame(x)
    df['label'] = y_pred
    df.to_csv(OUTPUT_FILE_NAME, header=False)
    plt.savefig(PLOT_FILE_NAME)


if __name__ == '__main__':
    args = sys.argv
    temp_func(args[1], int(args[2]))
