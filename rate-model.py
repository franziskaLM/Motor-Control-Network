import random
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as ss
import preparation
import initial_states


# System Parameters
N = 200 #number of neurons
n_ex = N / 2    #number of exhibitory neurons
n_in = N - n_ex #number of inhibitory neurons
r_0 = 5
r_max = 100
tau = 200  #time-constante in ms
simulation_time = 10000 # ms
delta_t = 10 #duration of a time-step


matrix = preparation.main(N, n_ex, n_in)
#initial_states = initial_states.main(N,matrix)
#print(initial_states)



#def matrix(N):
    #w_ex = 1
    #w_in = -1
    #matrix1 = scipy.sparse.random(N, int(N/2), density=0.1,data_rvs=np.ones)
    #matrix2 = scipy.sparse.random(N, int(N/2), density=0.1,data_rvs=np.ones)
    #matrix1 = np.array(matrix1.todense()) * w_ex
    #matrix2 = np.array(matrix2.todense()) * w_in
    #print(np.hstack((matrix1,matrix2)))
    #return np.hstack((matrix1,matrix2))

#matrix = matrix(N)

def ornstein_uhlenbeck():
    """
    Independent Ornstein-Uhlenbeck process for simulating the Noise-Input for each neuron

    Returns:
        Array with arrays in number of simulated time steps dt.
        Each array contains the noise value in number of Neurons for one time.

    """

    time_steps = int(simulation_time / delta_t) + 1 # number of time steps
    T_vec, dt = np.linspace(0, simulation_time, time_steps, retstep=True)

    kappa = 20  # mean reversion coefficient in Hz
    theta = 0  # long term mean in Hz
    sigma = 1.2
    #std_asy = np.sqrt(sigma ** 2 / (2 * kappa))  # asymptotic standard deviation 0.2Hz

    X0 = 0  # start value in Hz
    X = np.zeros((N, time_steps))
    X[:, 0] = X0
    W = ss.norm.rvs(loc=0, scale=1, size=(N, time_steps - 1))
    std_dt = np.sqrt(sigma ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))

    for t in range(0, time_steps - 1):
        X[:, t + 1] = theta + np.exp(-kappa * dt) * (X[:, t] - theta) + std_dt * W[:, t]

    return X



def create_state_space():   #TODO: state_space
    """

    Returns:
        random N-dim. vector, values betwenn (0,1)

    """
    v =  5 * np.random.rand(N,1)
    return v



class Network:


    def __init__(self):

        self.state_space = create_state_space()
        self.noise = ornstein_uhlenbeck()
       # self.initial_state
        self.delta_r = np.zeros((N,1))
        self.update_delta_r()
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
        #print("Noise:"+str(self.noise[simulation_step]))
        return self.noise[:,simulation_step]

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
        return (matrix @ self.delta_r) + self.get_noise()


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

    def update(self):
        self.update_steady_state()
        self.update_delta_r()
        self.update_time()


    def run(self):
        while self.time < simulation_time:
            self.update()

    def show_run(self):
        x_data = []
        y_data = []
        for unit in range(N):
            x_data.append([])
        while self.time <= simulation_time:
            for i in range(N):
                x_data[i].append(self.delta_r[i][0])
            y_data.append(self.time)
            self.update()
        for i in range(0,len(x_data)):
            plt.plot(y_data, x_data[i], label=str(unit))
        plt.show()


network = Network()
#network.run()
network.show_run()
