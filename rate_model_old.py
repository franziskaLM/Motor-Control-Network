import random

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as ss
import matrix_preparation
import initial_states

# System Parameters
N = 100  # number of neurons
n_ex = N / 2  # number of exhibitory neurons
n_in = N - n_ex  # number of inhibitory neurons
r_0 = 5
r_max = 100
tau = 200  # time-constante in ms
simulation_time = 1500  # ms
delta_t = 1  # duration of a time-step
state_number = 0

# Stimulus Parameter
t_go = 1000  # in ms
tau_before_go = 400  # in ms
tau_after_go = 2  # in ms

# initial systems
matrix = matrix_preparation.main(N, n_ex, n_in)
# initial_state = initial_states.main(N,matrix)[state_number]
a1 = initial_states.main(N, matrix)[0]
a2 = initial_states.main(N, matrix)[1]


def ornstein_uhlenbeck():
    """
    Independent Ornstein-Uhlenbeck process for simulating the Noise-Input for each neuron
    Returns:
        Array with arrays in number of simulated time steps dt.
        Each array contains the noise value in number of Neurons for one time.
    """

    time_steps = int(simulation_time / delta_t) + 1  # number of time steps
    T_vec, dt = np.linspace(0, simulation_time, time_steps, retstep=True)

    kappa = 20  # mean reversion coefficient in Hz
    theta = 0  # long term mean in Hz
    sigma = 1.5
    # std_asy = np.sqrt(sigma ** 2 / (2 * kappa))  # asymptotic standard deviation 0.2Hz

    X0 = 0  # start value in Hz
    X = np.zeros((N, time_steps))
    X[:, 0] = X0
    W = ss.norm.rvs(loc=0, scale=1, size=(N, time_steps - 1))
    std_dt = np.sqrt(sigma ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))

    for t in range(0, time_steps - 1):
        X[:, t + 1] = theta + np.exp(-kappa * dt) * (X[:, t] - theta) + std_dt * W[:, t]

    return X


######################################################################################


class Network:

    def __init__(self, initial_state, dot_product):

        self.state_space = -0.5 + np.random.rand(N, 1)
        self.noise = ornstein_uhlenbeck()
        self.delta_r = np.zeros((N, 1))
        self.update_delta_r()
        self.initial_state = initial_state
        self.matrix_dot_g = dot_product
        self.stimulus = self.initial_state - self.matrix_dot_g
        self.X = np.zeros((int(simulation_time/delta_t),N))
        self.time = 0

    def __str__(self):
        return f"Steady-State: {self.state_space}"

    def get_steady_state(self):
        """
        Returns:
            actual state space of the network
        """
        return self.state_space

    def get_time(self):
        """
        Returns:
            actual time-step of the network
        """
        return self.time

    def get_noise(self):
        """
        Returns:
            N-dim array with random independent noise-term for each neuron
        """
        simulation_step = int(self.time / delta_t)
        # print("Noise:"+str(self.noise[simulation_step]))
        return self.noise[:, simulation_step]

    def update_delta_r(self):
        """
        Returns:
            vector of the instantaneous singel-unit firing rate
        """
        for i, unit in enumerate(self.state_space):
            if unit[0] < 0:
                self.delta_r[i] = r_0 * math.tanh(unit[0] / r_0)
            else:
                self.delta_r[i] = (r_max - r_0) * math.tanh(unit[0] / (r_max - r_0))

    def get_input(self):
        """
        Returns:
            dot product of connectivity matrix * firing rate + constant input
        """
        return (matrix @ self.delta_r) + self.get_noise() + self.stimulus


    def get_delta_r(self):
        return self.delta_r

    def get_saved_data(self):
        return self.X


    def update_stimulus(self):
        if self.time < t_go:
            self.stimulus = math.exp(self.time / tau_before_go) * (self.initial_state - self.matrix_dot_g)
        else:
            self.stimulus = math.exp(- self.time / tau_after_go) * (self.initial_state - self.matrix_dot_g)

    def update_steady_state(self):
        """
        Returns:
            updated steady-state
        """
        self.state_space = self.state_space + delta_t * (-self.state_space / tau) + delta_t / tau * self.get_input()

    def update_time(self):
        """
        Returns:
            updated time step
        """
        self.time += delta_t

    def safe_data(self):
        for i, unit in enumerate(self.delta_r):
            self.X[self.time][i] = unit

    def update(self):
        self.safe_data()
        self.update_steady_state()
        self.update_delta_r()
        self.update_stimulus()
        # print(self.stimulus+self.state_space)
        self.update_time()

    def run(self):
        while self.time < simulation_time:
            self.update()

    def show_run(self):
        x_data = []
        y_data = []
        for unit in range(N):
            x_data.append([])
        while self.time < simulation_time:
            for i in range(N):
                x_data[i].append(self.delta_r[i][0])
            y_data.append(self.time)
            self.update()
        for i in range(0, len(x_data)):
            plt.plot(y_data, x_data[i], label=str(unit))
        plt.show()


####################################################################################
def calculate_g(condition):
    """
    Returns: gain function for calculation of the projection weights P(a) = a - W*g(a)
    depending on the initial state a
    """
    g = np.zeros((N, 1))
    for i, unit in enumerate(condition):
        if unit[0] < 0:
            g[i] = r_0 * math.tanh(unit[0] / r_0)
        else:
            g[i] = (r_max - r_0) * math.tanh(unit[0] / (r_max - r_0))
    return g


def matrix_dot_g(condition):
    """
    Returns: dot product of the connectivity matrix and the gain function g
    """
    g = calculate_g(condition)
    # print(condition)
    # print(g)
    return matrix @ g






def main():
    number_of_repetitions = 2
    all_X = []

    condition = random.choice([1, -1]) * random.uniform(0.5, 1) * np.array(a1) + random.choice(
        [1, -1]) * random.uniform(
        0.5, 1) * np.array(a2)
    for repeat in range(number_of_repetitions):
        network = Network(condition, matrix_dot_g(condition))
        network.show_run()
        all_X.append(network.get_saved_data())
    X = all_X[0]
    for elem in all_X[1:]:
        X = np.vstack((X,elem))
    return X




print(np.shape(main()))
