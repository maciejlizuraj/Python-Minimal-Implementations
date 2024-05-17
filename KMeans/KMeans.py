"""
This module implements the K-Means clustering algorithm for 2D data points.

Should be run with parameters: <input file> <output file> <image file> <number of clusters>

The `Points` class takes the following arguments:

* input_file_path (str): Path to the file containing the data points.
    The file should have two columns separated by delimiter from config and no header row.
    Column names are assumed to be 'X' and 'Y'.
* output_file_path (str): Path to save the clustered data points.
    The output file will have two columns: cluster assignment and data points (X, Y).
* image_file_path (str): Path to save the plot of the clustered data points.
* number_of_centroids (int): The desired number of clusters to generate.

The class performs the following steps:

1. Reads the data points from the file.
2. Initializes random centroids within the data range.
3. Assigns each data point to the closest centroid (repeatedly until convergence).
4. Recalculates the centroids based on the current cluster assignments.
5. Saves the clustered data points to a file.
6. Saves a plot of the data points and centroids colored by cluster assignment.

This module uses the following external libraries:

* pandas
* numpy
* matplotlib.pyplot
"""

import sys
from typing import Any

import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Config:
    """
    This class stores the configuration parameters for the K-Means clustering algorithm.

    Attributes:
        INPUT_FILE_PATH (str): Path to the file containing the data points.
        OUTPUT_FILE_PATH (str): Path to save the clustered data points.
        IMAGE_FILE_PATH (str): Path to save the plot of the clustered data points.
        NUMBER_OF_CENTROIDS (int): The desired number of clusters to generate.
        NUMBER_OF_ITERATIONS (int): The maximum number of iterations for the K-Means algorithm.
        CENTROID_MIN (numpy.ndarray): The minimum values for the centroid coordinates.
        CENTROID_MAX (numpy.ndarray): The maximum values for the centroid coordinates.
    """

    INPUT_FILE_PATH = ''
    OUTPUT_FILE_PATH = ''
    IMAGE_FILE_PATH = ''
    DELIMITER = '    '
    NUMBER_OF_CENTROIDS = 0
    NUMBER_OF_ITERATIONS = 10
    CENTROID_MIN = np.asarray((0, 0))
    CENTROID_MAX = np.asarray((0, 0))

    @staticmethod
    def read_file(input_file_path: str, output_file_path: str, image_file_path: str,
                  number_of_centroids: int) -> npt.NDArray[Any]:
        """
        Reads the data points from a file and sets the configuration parameters.

        Args:
            input_file_path (str): Path to the file containing the data points.
            output_file_path (str): Path to save the clustered data points.
            image_file_path (str): Path to save the plot of the clustered data points.
            number_of_centroids (int): The desired number of clusters to generate.

        Returns:
            numpy.ndarray: An array of shape (number_of_data_points, 2) containing the data points.
        """

        Config.INPUT_FILE_PATH = input_file_path
        Config.OUTPUT_FILE_PATH = output_file_path
        Config.IMAGE_FILE_PATH = image_file_path
        Config.NUMBER_OF_CENTROIDS = number_of_centroids
        df = pd.read_csv(Config.INPUT_FILE_PATH, sep='    ', header=None, names=['X', 'Y'],
                         engine='python')
        points = df[['X', 'Y']].to_numpy()

        min_values = np.amin(points, axis=0)
        max_values = np.amax(points, axis=0)

        Config.CENTROID_MIN = min_values
        Config.CENTROID_MAX = max_values

        return points


class Points:
    """
        This class performs K-Means clustering on a set of 2D points.

        Args:
            input_file_path (str): Path to the file containing the data points.
                The file should have two columns separated by delimiter from config and no header
                row. Column names are assumed to be 'X' and 'Y'.
            output_file_path (str): Path to save the clustered data points.
                The output file will have two columns: cluster assignment and data points (X, Y).
            image_file_path (str): Path to save the plot of the clustered data points.
            number_of_centroids (int): The desired number of clusters to generate.

        Attributes:
            centroids (numpy.ndarray): An array of shape (number_of_centroids, 2) containing the
            centroids of each cluster.
            assignments (numpy.ndarray): An array of shape (number_of_data_points, 1) containing
            the cluster assignment for each data point.
            points (numpy.ndarray): An array of shape (number_of_data_points, 2) containing the
            coordinates of the data points. Each row represents the (X, Y) coordinates of a data
            point.
            colormap (numpy.ndarray): An array of shape (number_of_centroids, 3) containing random
            colors assigned to each centroid. Each row represents the RGB color (red, green, blue)
            of a centroid used for visualization.
        """

    def __init__(self, input_file_path: str, output_file_path: str, image_file_path: str,
                 number_of_centroids: int):
        self.centroids = np.array([0, 0])
        self.assignments = np.array([0])
        self.points = Config.read_file(input_file_path, output_file_path, image_file_path,
                                       number_of_centroids)
        self.colormap = np.random.rand(Config.NUMBER_OF_CENTROIDS, 3)

        self.generate_random_centroids()
        self.assign_points_to_centroids()

        while Config.NUMBER_OF_ITERATIONS > 0:
            self.recalculate_centroids()
            self.assign_points_to_centroids()
            Config.NUMBER_OF_ITERATIONS -= 1

        np.savetxt(Config.OUTPUT_FILE_PATH, np.c_[self.assignments, self.points], delimiter=' ')
        self.save_plot()

    def generate_random_centroids(self) -> None:
        """
        Generates random initial centroids within the data point range.

        This method initializes the `centroids` attribute with random positions within the
        valid range of the data points. It uses the `Config.CENTROID_MIN` and
        `Config.CENTROID_MAX` attributes to define the boundaries for generating
        random centroid coordinates.
        """
        self.centroids = np.random.rand(Config.NUMBER_OF_CENTROIDS, 2) * (
                Config.CENTROID_MAX - Config.CENTROID_MIN) + Config.CENTROID_MIN

    def assign_points_to_centroids(self) -> None:
        """
        Assigns each data point to the closest centroid.

        This method calculates the Euclidean distance between each data point and all centroids.
        It then assigns each data point to the cluster of the closest centroid. The assignments
        are stored in the `assignments` attribute.
        """
        distances_tmp = []
        for centroid in self.centroids:
            dist = np.sqrt(
                (self.points[:, 0] - centroid[0]) ** 2 + (self.points[:, 1] - centroid[1]) ** 2)
            distances_tmp.append(dist)
        distances = np.array(distances_tmp)
        self.assignments = np.argmin(distances, axis=0)
        self.assignments = self.assignments.reshape(self.points.shape[0], 1)

    def assign_colors_to_points(self) -> npt.NDArray[Any]:
        """
        Assigns a color from the colormap to each data point based on its cluster assignment.

        This method uses the `assignments` attribute to index into the `colormap` attribute
        and retrieve the corresponding color for each data point. The colors are returned as a
        NumPy array.
        """
        colors = self.colormap[self.assignments]
        return colors.reshape(colors.shape[0], colors.shape[2])

    def save_plot(self) -> None:
        """
        Saves the scatter plot of the data points and centroids to an image file.

        This method is similar to `show_plot` but instead of displaying the plot, it saves it
        as an image file. It uses the `matplotlib.pyplot` library to create the plot and then
        saves it to the file path specified by the `Config.IMAGE_FILE_PATH` attribute using
        `plt.savefig`.
        """
        colors = self.assign_colors_to_points()
        plt.scatter(self.points[:, 0], self.points[:, 1], c=colors, alpha=0.2)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c=self.colormap)
        plt.savefig(Config.IMAGE_FILE_PATH)

    def recalculate_centroids(self) -> None:
        """
        Recalculates the centroids based on the current cluster assignments.

        This method iterates over the clusters and calculates the average position (mean) of
        all data points assigned to that cluster. The new centroids are then stored in the
        `centroids` attribute. This process is repeated for a specified number of iterations
        (controlled by `Config.NUMBER_OF_ITERATIONS`).
        """
        new_centroids = []
        for idx, old_centroid in enumerate(self.centroids):
            assignment = self.assignments == idx
            assignment = assignment.flatten()
            cluster = self.points[assignment, :]
            if len(cluster) == 0:
                new_centroids.append(old_centroid)
            else:
                new_centroids.append(np.mean(cluster, axis=0))
        self.centroids = np.asarray(new_centroids)


if __name__ == '__main__':
    args = sys.argv
    Points(args[1], args[2], args[3], int(args[4]))
