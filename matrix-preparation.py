import numpy as np
import numpy.linalg as linalg
import scipy.sparse
from scipy.sparse import coo_array
from scipy import linalg
import random
import math
import matplotlib.pyplot as plt


# Connectivity attributes
R = 10    #spectral radius
p = 0.1  # connectivity density of exhibi. neurons
p_in = 0.4  # connectivity density of inhibi. neurons
gamma = 3
omega = (math.sqrt(2) * R) / (math.sqrt(p * (1 - p) * (1 + (gamma ** 2))))
eta = 10  # learning rate


### Create initially Connectivity Matrix ###


def all_indices(size: tuple) -> list:
    """

    Args:
        size: tuple of matrix dimensions

    Returns:
        list of all matrix indices

    """
    first_dim = int(size[0])
    second_dim = int(size[1])
    indices_list = []
    for i in range(first_dim):
        for j in range(second_dim):
            if i != j:
                indices_list.append((i, j))
    return indices_list


def filled(matrix_row: list, size: tuple, dens: float) -> bool:
    """

    Args:
        matrix_row: list with row-coordinates
        size: tuple with matrix dimension
        dens: float of the connectivity density

    Returns:
        helperfunction for testing connectivity density of matrix
    """
    first_dim = size[0]
    second_dim = size[1]
    if (len(matrix_row) / (first_dim * second_dim)) >= dens:
        return True
    else:
        return False


def initially_entries(size, dens, weight, free) -> tuple:
    """

    Args:
        size: tuple of the matrix (-block) dimension
        dens: density of matrix block
        weight: start value
        free: list with free indices

    Returns:
        Tuple with coo-matrix and list with free indices in this matrix
    """
    row = []
    col = []
    val = []
    while not filled(row, size, dens):
        x = random.randint(0, len(free) - 1)
        new_connection = free[x]
        row.append(free[x][0])
        col.append(free[x][1])
        val.append(weight)
        free.remove(free[x])
    return (row, col, val), free


def exhibi(N, n_ex) -> tuple:
    """
        return: tuple of coo-matrix (row, columns, values) and list with free indices
    """
    size = (N, n_ex)
    w_ex = omega / math.sqrt(N)
    (block_ex, free_ex) = initially_entries(size, p, w_ex, all_indices(size))
    return block_ex, free_ex


def inhibi(N, n_in):
    """
        return: tuple of coo-matrix (row, columns, values) and list with free indices
    """
    size = (N, n_in)
    w_in = -1 * gamma * omega / math.sqrt(N)
    (block_in, free_in) = initially_entries(size, p_in, w_in, all_indices(size))
    return block_in, free_in


def sparse_matrix(exhibi_block, inhibi_block, N, n_ex, n_in):
    """

    Args:
        exhibi_block: list with three entries [row,column,values] for creating a coo-matrix
        inhibi_block: list with three entries [row,column,values] for creating a coo-matrix

    Returns:
        Connectivity Matrix in sparse-shape
    """
    coo_inhibi = coo_array((inhibi_block[2], (inhibi_block[0], inhibi_block[1])),
                           shape=(int(N), int(n_in)))  # coo_array(data,(rows,columns))
    coo_exhibi = coo_array((exhibi_block[2], (exhibi_block[0], exhibi_block[1])), shape=(N, n_ex))
    w = scipy.sparse.hstack([coo_exhibi, coo_inhibi])
    return w


def sparse_to_array(matrix):
    a = matrix.toarray()
    return a


### Calculate spectral Abscissa ###
def eigenvalue(matrix):
    """

    Args:
        matrix: Input Matrix

    Returns:
        returns the eigenvalues ew of matrix
    """
    ew, *_ = linalg.eig(matrix)  # return (eigenvalue,eigenvektor)
    return ew


def spectral_abscissa(matrix):
    """

    Args:
        matrix: Input Matrix

    Returns:
        return of the spectral abscissa
        that is the greatest real part of the matrix A's spectrum
    """
    real_ew = np.real(eigenvalue(matrix))  # Re(ew)
    max_real_ew = np.max(real_ew)  # Max(Re(ew))

    return max_real_ew


def smoothed_sa(matrix):
    """

    Args:
        matrix: input matrix

    Returns:
        return of the 'smoothed-spectral abscissa'
        that is defined by the 1.5 fold of the spectral abscissa sa
    """
    sa = spectral_abscissa(matrix)
    if 1.5 * sa > sa + 0.2:
        return 1.5 * sa
    else:
       return sa + 0.2


########## Helper Functions for calculation of the Gradient ########

def shifted_matrix(matrix, ssa, N):
    """
    Args:
        matrix: input matrix
        sa: spectral abscissa

    Returns:
       diagonal N x N Matrix ;
        main diagonal filled with value of the smoothed spectral abscissa
    """
    diag = [ssa] * N
    return np.diag(diag)


def shifter(A, ssa, N):
    """
    Args:
        matrix: input matrix
        sa: spectral abscissa

    Returns:
       return of the shifted Matrix (A-shift*1)
    """
    return A - shifted_matrix(A, ssa, N)


def solve_lyapunov(A, N):
    """

    Args:
        A: Input Matrix

    Returns:
            return Solution to the continuous Lyapunov equation
            B*X + X*B_trans = Z

    """
    diag_Z = [-2] * N
    Z = np.diag(diag_Z)  # diagonal N x N Matrix (-2*I)

    return scipy.linalg.solve_continuous_lyapunov(A, Z)


def calculate_Q(ws, N):
    """
    Args:
        A: Input Matrix
        ws: shifted matrix

    Returns:
        return N x N matrix Q
        that is Solution to the equation ( ws_trans*Q + Q*ws = -2*I)

    """

    ws_trans = np.transpose(ws)  # transposed matrix ws
    return solve_lyapunov(ws_trans, N)


def calculate_P(ws, N):
    """
    Args:
        A: Input Matrix
        ws: shifted matrix

    Returns:
        return N x N matrix P
        that is Solution to the equation ( ws*Q + Q*ws_trans = -2*I)

    """
    return solve_lyapunov(ws, N)


def gradient(A, ssa, N):
    """

    Args:
        A: Input Matrix
        sa: spectral abscissa

    Returns:
        gradient for minimization of the smoothed abscissa

    """
    ws = shifter(A, ssa, N)
    Q = calculate_Q(ws, N)
    P = calculate_P(ws, N)
    grad = Q @ P / np.trace(Q @ P)  # gradient

    return grad


### Matrix Optimization

def enforce_negativity(block, free, x):
    """

    Args:
        block: coo-matrix
        free: list with free indices
        x: indices

    Returns:
    setting a non-negative value to zero and creating a new connectivity
    """
    row = block[0]
    col = block[1]
    weights = block[2]

    new_index = free.pop(0)
    old_index = (row.pop(x), col.pop(x))
    weights.pop(x)
    weights.append(0)
    row.append(new_index[0])
    col.append(new_index[1])
    free.append(old_index)

def plotter(matrix):
    ew, ev = linalg.eig(matrix)
    real = np.real(ew)
    imag = np.imag(ew)
    plt.plot(real, imag, "ob")
    plt.show()

def move_weights(exhibi_block, inhibi_block, free_inhibi, old_sa, counter, N, n_ex, n_in):
    """

    Args:
        exhibi_block: coo-matrix of exhibi block
        inhibi_block: coo-matrix of inhibi block
        free_inhibi: list with indices of free entries in inhibi-block
        old_sa: last spectral abscissa
        counter: number of occurrence of a value of the spectral abscissa

    Returns:
        recursive function that shifted the values of the connectivity matrix till spectral abscissa is converging

    """
    A = sparse_to_array(sparse_matrix(exhibi_block, inhibi_block, N, n_ex, n_in))  # connectivity matrix (array)
    ssa = smoothed_sa(A)  # smoothed spectral abscissa
    grad = gradient(A, ssa, N)
    row = inhibi_block[0]
    col = inhibi_block[1]
    weights = inhibi_block[2]
    #print("smoothed spectral abscissa:"+str(ssa))
    if ssa <= 0.81:
        #print(A)
        return A
    else:
        x = 0
        while x < len(weights):
            weight = weights[x]
            i = row[x]
            j = col[x]
            weight = weight - eta * grad[i][j + n_ex]  # gradient-shifted weight
            if weight >= 0:
                enforce_negativity(inhibi_block, free_inhibi, x)
            else:
                weights[x] = weight
            x += 1
        return move_weights(exhibi_block, (row, col, weights), free_inhibi, old_sa, counter, N, n_ex , n_in)


def main(N, n_ex, n_in):

    """

    Returns:
        optimized connectivity matrix

    """
    # create left side of connectivity matrix
    exh = exhibi(N, n_ex)[0]
    # create right side of connectivity matrix
    inh, free = inhibi(N, n_in)
    return move_weights(exh, inh, free, 0, 0, int(N), int(n_ex), int(n_in))


#w=main(4, 2, 2)
#print(w)
#ew,ev = linalg.eig(w)
#real = np.real(ew)
#imag = np.imag(ew)
#plt.plot(real, imag,"ob")
# #plt.show()


