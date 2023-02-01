import random
import matplotlib.pyplot as plt
import numpy as np
import math

import scipy
import scipy.stats as ss
from numpy import linalg
import preparation
import initial_states

random.seed(42)


# System Parameter of the Rate Model


# creating Noise Input
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


#############################################################################

class Network:

    def __init__(self, X, count):

        self.state_space = -0.5 + np.random.rand(N, 1)  # state-space in size (N,1)
        self.delta_r = np.zeros((N, 1))  # change of the firing rate in Hz in size (N,1)
        self.update_delta_r()  # initialized delta_r depending on state_space
        self.time = 0  # time in ms
        self.stimulus = initial_state - dot_product  # external stimulus in size(N,1)

        self.noise = ornstein_uhlenbeck()  # creating noise-input in size (N,T)

        # self.X = np.zeros((int(simulation_time / delta_t), N))

        self.X = X
        self.actual_state = count

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
            vector of the instantaneous single-unit firing rate

        """
        for i, unit in enumerate(self.state_space):
            if unit[0] < 0:
                self.delta_r[i] = r_0 * math.tanh(unit[0] / r_0)
            else:
                self.delta_r[i] = (r_max - r_0) * math.tanh(unit[0] / (r_max - r_0))

    def get_input(self):
        """

        Returns:
            dot product of connectivity matrix * firing rate + noise + stimulus

        """
        return (matrix @ self.delta_r) + self.get_noise()  # + self.stimulus #TODO

    def update_stimulus(self):
        """

        Returns:
            stimulus-vector - depending if point of time is before (exp. rise) OR after (exp.decay) go cue

        """
        if self.time < t_go:
            self.stimulus = math.exp(self.time / tau_before_go) * (initial_state - dot_product)
        else:
            self.stimulus = math.exp(- self.time / tau_after_go) * (initial_state - dot_product)

    def update_X(self):
        """

        Returns:
            appends current firing-rate to firing-rate collection

        """
        for n in range(N):
            # print(int(self.time / delta_t)+self.actual_state*int(simulation_time / delta_t))
            self.X[int(self.time / delta_t) + self.actual_state * int(simulation_time / delta_t)][n] = self.delta_r[n][
                0]

    def update_state_space(self):
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

    def update(self):
        """

        Call for update

        """
        self.update_X()
        self.update_state_space()
        self.update_delta_r()
        self.update_stimulus()
        self.update_time()

    def run(self):
        """

        makes simulation running

        """
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

    def get_X(self):
        return self.X


##################################################################


def calculate_weights(X, Z):
    """

    Args:
        X: collection of all firing rates for each time over simulation time in size (T, N)
        Z: definite trajectory-vector in size(T,1)

    Returns:
        readout weights (m1,m2) optimized through least-squares regression

    """
    return ((scipy.linalg.pinv((X.T @ X)) @ X.T @ Z[0]).T, (scipy.linalg.pinv((X.T @ X)) @ X.T @ Z[1]).T)


##########################################################################

# Helper Functions

def calculate_g(condition):
    """

    Returns: gain function for calculation of the projection weights P(a) = a - W*g(a)
    depending on the initial state a
    """
    g = np.zeros((N, number_of_states))
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


def save_fire_rates():
    """
        Makes the network run for each initial condition b and saves the fire-rates for every time step
    Returns:
        array in size( T * B , N)
        where T: number of time steps
              b: number of initial states b
              N: Number of Neurons

    """

    # print()
    X = np.zeros((len(initial_conditions) * int(simulation_time / delta_t) * number_of_repetitions, N))
    i = 0
    for count, condition in enumerate(initial_conditions):
        for repeat in range(number_of_repetitions):
            global initial_state
            global dot_product

            initial_state = condition

            dot_product = matrix_dot_g(condition)

            network = Network(X, i)
            network.run()

            X = network.get_X()
            i += 1
    return X


def run_and_safe_data(matrixx, int_cond, parameter):
    """

    Args:
        matrixx: connectivity-weights in size (N,N)
        trajectories: list with end-coordinates of all trajectroies
        int_cond: tupel with two initial condition (a1, a2)

    Returns: makes Network run for each trajectory with a different initial-condition and returns the saved fire rates
    and linear trajectories

    """
    global matrix
    global initial_conditions
    global N, n_ex, n_in, r_0, r_max, tau, simulation_time, delta_t, number_of_states, t_go, tau_before_go, tau_after_go
    global number_of_repetitions
    N = parameter["N"]
    n_ex = parameter["n_ex"]  # number of exhibitory neurons
    n_in = parameter["n_in"]  # number of inhibitory neurons
    r_0 = parameter["r_0"]  # base rate in Hz
    r_max = parameter["r_max"]  # max. rate in Hz
    tau = parameter["tau"]  # time - const of neuron membran in ms
    simulation_time = parameter["simulation_time"]  # in ms
    delta_t = parameter["delta_t"]  # duration of a time-step

    # Stimulus Parameter
    number_of_states = parameter["number_of_states"]
    t_go = parameter["t_go"]  # point of time of the go cue # in ms
    tau_before_go = parameter["tau_before_go"]  # time const of the rise during preparation time
    tau_after_go = parameter["tau_after_go"]  # time const of the decay after go cue
    number_of_repetitions = parameter["number_of_repetitions"]  # number of repetitions  #TODO: Seed?

    matrix = matrixx
    initial_conditions = int_cond

    X = save_fire_rates()
    return X

# print(main())
# print(main())
