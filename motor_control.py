import preparation
import numpy as np
import random
import least_squares_regression
import initial_states
import rate_model

# System Parameter of the Rate Model
N = 20  # number of neurons
n_ex = N / 2  # number of exhibitory neurons
n_in = N - n_ex  # number of inhibitory neurons
r_0 = 5  # base rate in Hz
r_max = 100  # max. rate in Hz
tau = 200  # time - const of neuron membran in ms
simulation_time = 10000  # in ms
delta_t = 1  # duration of a time-step

# Stimulus Parameter
state_number = 0  # set the initial state, e.g. a0
number_of_states = 2
t_go = 1000  # point of time of the go cue # in ms
tau_before_go = 400  # time const of the rise during preparation time
tau_after_go = 2  # time const of the decay after go cue

# trajectory params
end_coord = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

# train system
number_of_repetitions = 10

# initial systems
matrix = preparation.main(N, n_ex, n_in)  # optimized connectivity matrix

params = {"N": N, "n_ex": n_ex, "n_in": n_in, "r_0": r_0, "r_max": r_max, "tau": tau,
          "simulation_time": simulation_time, "delta_t": delta_t,
          "number_of_states": number_of_states, "t_go": t_go, "tau_before_go": tau_before_go,
          "tau_after_go": tau_after_go,
          "number_of_repetitions": number_of_repetitions}


def create_initial_conditions(a1, a2):
    """

    Args:
        a1: initial state a1
        a2: initial state a2

    Returns:
        array with random initial states formed by linear combination from a1 and a2

    """
    states = []
    # create for each trajectorie one random state condition b
    for k in range(len(end_coord)):
        states.append(
            random.choice([1, -1]) * random.uniform(0.5, 1) * np.array(a1) + random.choice([1, -1]) * random.uniform(
                0.5, 1) * np.array(a2))
    return (np.array(states))


def linear_trajectory():
    """
    creates linear trajectory for each given end-coordinate with start in (0,0)
    Returns: list with tuples of (x_coordinates, y_coordinates)

    """
    trajectories_x = []  # array with x-coord of all trajectories
    trajectories_y = []  # array with y-coord of all trajectories
    for count, direction in enumerate(end_coord):
        x_coord = np.zeros((int((simulation_time - t_go) / delta_t), 1))
        y_coord = np.zeros((int((simulation_time - t_go) / delta_t), 1))
        end_x = direction[0]
        end_y = direction[1]
        step_length_x = end_x / int((simulation_time - t_go) / delta_t)
        step_length_y = end_y / int((simulation_time - t_go) / delta_t)
        for step in range(1, (int((simulation_time - t_go) / delta_t))):
            x_coord[step] = x_coord[step - 1] + step_length_x
            y_coord[step] = y_coord[step - 1] + step_length_y
        trajectories_x.append(x_coord)
        trajectories_y.append(y_coord)
    return (trajectories_x, trajectories_y)


def assignment(trajectories, initial_conditions):
    """

    Args:
        trajectories: array with n trajectories for each end-coord
        initial_conditions: array with n initial conditions b_i

    Returns: list with tuples where trajectories are assigned to one initial condition

    """
    pair = []
    for i, trajectory in enumerate(trajectories):
        pair.append((trajectory, initial_conditions[i]))
    return pair


def create_b(b1, b2):
    """

    Args:
        b1: bias for x-dim
        b2: bias for y-dim

    Returns:
         array in size(2,T) where T: number of time-steps


    """
    b1 = np.ones((1, int(simulation_time - t_go / delta_t))) * b1[0]
    b2 = np.ones((1, int(simulation_time - t_go / delta_t))) * b2[0]
    return np.vstack((b1, b2))


def create_m(m1, m2):
    return np.vstack((m1.T, m2.T))


def ttry(pairs, m1, b1, m2, b2):
    params["number_of_repetitions"] = 1
    condition = pairs[0][1]
    delta_r = rate_model.run_and_safe_data(matrix, [condition], params).T[:, int(t_go/delta_t):]
    # print(np.shape(delta_r))
    b = create_b(b1, b2)
    m = create_m(m1, m2)
    return m @ delta_r #+ b


trajectories = linear_trajectory()
initial_conditions = create_initial_conditions(initial_states.main(N, matrix)[0], initial_states.main(N, matrix)[1])
pairs = assignment(trajectories, initial_conditions)
X = rate_model.run_and_safe_data(matrix, initial_conditions, params)[t_go * number_of_repetitions * len(end_coord):, :]
#print(np.shape(X))
m1, b1, m2, b2 = least_squares_regression.main(X, trajectories, params)
print(ttry(pairs, m1, b1, m2, b2))
