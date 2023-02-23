# Motor-Contol-Network
This is an optimized, stable network that allows learned movements to be learned and executed in the form of trajectories. 

For this purpose, an optimized connectivity matrix is generated in the module "matrix-preparation.py", which allows stable dynamics. This is realized by minimizing the smoothed spectral abscissa.

In the module "initial-states" the preferred initial states of the network are determined, which maximize the energy of the network.

The module "rate-model.py" contains the implementation of a rate-based unit network as a class, which simulates the firing rates of the network for discrete time steps. The network is implemented as a class and returns a matrix that stores all firing rates for each invoked condition and trial.

The module "linear-regression.py" is used to train the network. By specifying an optimal movement in the form of a trajectory z, the projection weights for a linear model are created, which enables the determination of the executed movement of a network.

The module "Main_motor_contol" uses these modules to train and execute movements.
