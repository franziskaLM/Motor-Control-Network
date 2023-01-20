import preparation
import numpy as np
import scipy
import scipy.integrate
from scipy import linalg
from operator import itemgetter


def solve_lyapunov(matrix, N):
    A = matrix - np.identity(int(N))
    B = -2 * np.identity(int(N))
    return scipy.linalg.solve_continuous_lyapunov(np.transpose(A), B)

def get_eigenvectors(Q):
    return linalg.eig(Q)

def calculate_energy(ev,Q,N):
    states = []
    for a in ev:
        energy = np.transpose(a) @ Q @ a
        states.append((energy,np.reshape(a,(N,1))))
    return states

def get_sorted_stats(ev, Q,N):
    states_and_energies = calculate_energy(ev,Q,N)
    states_and_energies.sort(key=itemgetter(0), reverse=True)
    only_states = []
    for elem in states_and_energies:
        only_states.append(elem[1])     #append only vector
    return only_states


def main(N, matrix):
    Q = solve_lyapunov(matrix, N)
    ew,ev = get_eigenvectors(Q)
    sorted_stats = get_sorted_stats(ev, Q,N)
    return sorted_stats

#print("Only states "+str(main(int(10),preparation.main(int(10), int(5), int(5)))))
