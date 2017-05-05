"""
Functions which will be used in optimization methods.
The 'thetas' argument will always be an n times s 2-d numpy array.
If you need this ta be a vector instead, take in the matrix and
perform thetas.reshape(n*s, 1) instead.
"""
import numpy as np

from constraints import (
    generate_constraints_function,
    generate_constraint_gradients_function,
)

def generate_objective_function(robot_arm):
    n = robot_arm.n
    s = robot_arm.s

    def objective(thetas):
        if not thetas.shape == (n*s,):
            raise ValueError('Thetas not given as a single 1D-vetor, but as: ' + str(thetas.shape))

        rotated = np.roll(thetas.copy().reshape((n, s), order='F'), shift=-1, axis=1)
        deltas = rotated - thetas.reshape((n, s), order='F')
        return 0.5 * np.sum(deltas ** 2)

    return objective

def generate_objective_gradient_function(robot_arm):
    n = robot_arm.n
    s = robot_arm.s

    def objective_gradient(thetas):
        if not thetas.shape == (n*s,):
            raise ValueError('Thetas not given as a single 1D-vetor, but as: ' + str(thetas.shape))
        else:
            thetas = thetas.reshape((n, s), order='F')

        def roll(x, y): return np.roll(x.copy(), shift=y, axis=1)

        return (2*thetas - roll(thetas, -1) - roll(thetas, 1)).reshape((n*s,), order='F')

    return objective_gradient
