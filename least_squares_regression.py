import numpy as np
from numpy.linalg import linalg
import math
import random


def zip_of(trajectories):
    """

    Args:
        trajectories: array with x- (or y-) coordinates for all trajectories in every given direction

    Returns: create an array with consideration of the repetitions in size ( T * directions * repetitions, 1)
            where T: number of time-steps
                  directions : number of trajectories
                  repetitions: repeat of the executation each trajectory

    """
    all_coordinates = np.zeros((len(trajectories) * number_of_repetitions * int((simulation_time - t_go) / delta_t), 1))
    i = 0
    for trajectory in trajectories:
        for repeat in range(number_of_repetitions):
            for coord in trajectory:
                all_coordinates[i] = coord[0]
                i += 1
    return all_coordinates


def create_Z(x_coord, y_coord):
    Z1 = zip_of(x_coord)
    Z2 = zip_of(y_coord)
    return (Z1, Z2)


def least_squares_regression(X, Z):
    return linalg.pinv(X.T @ X) @ X.T @ Z


def main(X, all_Z, parameter):
    """

    Args:
        X: saved Fire-rates for all initial conditions and each time step
        all_Z: trajectories for each condition, seperatly saved in lists for each coordinate (x resp. y)
        parameter: dict with parameters

    Returns: optimized readout weights

    """
    global simulation_time, delta_t, number_of_states, t_go, tau_before_go, tau_after_go
    global number_of_repetitions
    N = parameter["N"]
    simulation_time = parameter["simulation_time"]  # in ms
    delta_t = parameter["delta_t"]  # duration of a time-step
    t_go = parameter["t_go"]
    # Stimulus Parameter
    number_of_states = parameter["number_of_states"]
    number_of_repetitions = parameter["number_of_repetitions"]  # number of repetitions  #TODO: Seed?

    size_X = np.shape(X)
    X = np.hstack((X, np.ones((size_X[0], 1))))

    Z_x, Z_y = create_Z(all_Z[0], all_Z[1])  # trajectories of the x-coord. rep. y-coord

    readout_weights = (least_squares_regression(X, Z_x), least_squares_regression(X, Z_y))
    m1 = readout_weights[0][:N, :]
    m2 = readout_weights[1][:N, :]
    b1 = readout_weights[0][N:, :]
    b2 = readout_weights[1][N:, :]
    return (m1, b1, m2, b2)
