"""
This is the main program that can be used to generate, train and run a complete network
"""
import numpy as np
import random
import least_squares_regression
import initial_states
import rate_model
import matrix_preparation
import matplotlib.pyplot as plt

# System Parameter of the Rate Model
N = 100  # number of neurons
n_ex = N / 2  # number of exhibitory neurons
n_in = N - n_ex  # number of inhibitory neurons
r_0 = 5  # base rate in Hz
r_max = 100  # max. rate in Hz
tau = 200  # time - const of neuron membran in ms
simulation_time = 2000  # in ms
delta_t = 1  # duration of a time-step

# Stimulus Parameter
state_number = 0  # set the initial state, e.g. a0
t_go = 1000  # point of time of the go cue # in ms
tau_before_go = 400  # time const of the rise during preparation time
tau_after_go = 2  # time const of the decay after go cue

# trajectory params
end_coord = [(1, 1)]  # , (-1, 1), (-1, -1), (1, -1)]
numbers_of_trajectories = len(end_coord)

# train system
number_of_repetitions = 5   #number of trials

#dicterory that stores all parameters
params = {"N": N, "n_ex": n_ex, "n_in": n_in, "r_0": r_0, "r_max": r_max, "tau": tau,
          "simulation_time": simulation_time, "delta_t": delta_t,
          "t_go": t_go, "tau_before_go": tau_before_go,
          "tau_after_go": tau_after_go,
          "number_of_repetitions": number_of_repetitions,
          "number_of_trajectories": numbers_of_trajectories}

def butterfly_trajectory():
    """

    Returns:
        Example Trajectory: butterfly trajectory

    """
    t = np.linspace(0, 2 * np.pi, int((simulation_time - t_go) / delta_t))
    x = np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5)
    y = np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5)
    return ([x], [y])

def create_initial_conditions(a1, a2):
    """
    Args:
        a1: initial state a1 (preferred initials state)
        a2: initial state a2 (second preferred initial state)
    Returns:
        array with random initial states formed by linear combination from a1 and a2
    """
    states = []
    # create for each trajectory one random state condition b
    for k in range(numbers_of_trajectories):
        states.append(np.array(a2))
            random.choice([1, -1]) * random.uniform(0.5, 1) * np.array(a1) + random.choice([1, -1]) * random.uniform(
               0.5, 1) * np.array(a2))
    return (np.array(states))



def create_b(b1, b2):
    """
    Args:
        b1: bias for x-dim
        b2: bias for y-dim
    Returns:
         biases of the linear regression
         array in size(2,T) where T: number of time-steps
    """
    b1 = np.ones((1, int(simulation_time - t_go / delta_t))) * b1[0]
    b2 = np.ones((1, int(simulation_time - t_go / delta_t))) * b2[0]
    return np.vstack((b1, b2))


def create_m(m1, m2):
    """
    
    Args:
        m1: projection weights for x-coord
        m2: projection weights for y-coord

    Returns:
        projection weights of the linear regression

    """
    return np.vstack((m1.T, m2.T))


def paint_trajectory(m1, b1, m2, b2, condition):
    """
    :param m1: opt. readout weights x coord
    :param b1: const. bias x-coord
    :param m2: opt. readout weights y coord
    :param b2: const bias y coord
    :param condition: initial condition
    :return: trajectory for initial condition
    """
    #set number of trials to 1
    params["number_of_repetitions"] = 1
    
    #run the network with the given condition as stimulus
    delta_r = rate_model.safe_data(matrix, [condition], params).T[:, int(t_go / delta_t):] 
    
    #calculate projection weiht and biases wit Normal Equation
    b = create_b(b1, b2)
    m = create_m(m1, m2)
    
    #linear regression
    z1, z2 = m @ delta_r + b
    return (z1, z2)


def paint_all_in_one(m1, b1, m2, b2, conditions):
    """
    
    Args:
        :param m1: opt. readout weights x coord
        :param b1: const. bias x-coord
        :param m2: opt. readout weights y coord
        :param b2: const bias y coord
        :param condition: initial condition

    Returns:
        plot function - to plot all executed movements in one plot

    """
    for condition in conditions:
        z1, z2 = paint_trajectory(m1, b1, m2, b2, condition)
        plt.plot(z1, z2,c=colour,linewidth=2)
    plt.show()


def train(matrix, conditions, params):
    """
    
    Args:
        matrix: connectivity matrix
        conditions: all initial condition that should be use for stimulating 
        params: dict with stored parameters

    Returns:
        matrix X with stored firing rates for each trial and condition

    """
    return rate_model.safe_data(matrix, conditions, params)


def main():
    """
    Main Function:
        +creates connectivity matrix with modul matrix_preparation
        +computes initial conditions for each trained movement (for each trajectory in trajectories)
        +training by storing firing rates for each condition in matrix X

    """
    global matrix
    matrix = matrix_preparation.main(N, n_ex, n_in)  # optimized connectivity matrix
    
    #determine trajectory of the movement
    trajectories = butterfly_trajectory() 
    
    #create initial conditions for each trajectory
    initial_conditions = create_initial_conditions(initial_states.main(N, matrix)[0], initial_states.main(N, matrix)[1])
    
    #run the network for each condition and trial and safe all firing rates in matrix X
    X = train(matrix, initial_conditions, params)
    
    #compute projection weights for regression by Normal Equation
    m1, b1, m2, b2 = least_squares_regression.main(X, trajectories, params)
    
    #plot the executete movements
    paint_all_in_one(m1, b1, m2, b2, initial_conditions)


