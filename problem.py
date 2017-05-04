"""
Functions which will be used in optimization methods.
The 'thetas' argument will always be an n times s 2-d numpy array.
If you need this ta be a vector instead, take in the matrix and
perform thetas.reshape(n*s, 1) instead.
"""
import numpy as np
import functools


def objective(thetas):
    assert isinstance(thetas, np.ndarray)
    rotated = np.roll(thetas.copy(), shift=-1, axis=1)
    deltas = rotated - thetas
    return 0.5 * np.sum(deltas ** 2)


def constraint(thetas, lengths, constraint_number, coordinate):
    n = len(lengths)
    col_index = None
    if constraint_number % 2 == 1:
        col_index = (constraint_number + 1) // 2 - 1
    elif constraint_number % 2 == 0:
        col_index = constraint_number // 2 - 1
    theta_cum_sum = np.cumsum(thetas[:, col_index])
    constraint_value = 0
    for i in range(n):
        if constraint_number % 2 == 0:
            constraint_value += lengths[i] * np.sin(theta_cum_sum[i])
        elif constraint_number % 2 == 1:
            constraint_value += lengths[i] * np.cos(theta_cum_sum[i])
    return constraint_value - coordinate


def objective_gradient(thetas):
    assert isinstance(thetas, np.ndarray)
    n, s = thetas.shape
    objective_gradient_matrix = np.zeros((n, s))
    for j in range(s):
        for i in range(n):
            if (j > 0) and j < (s - 1):
                objective_gradient_matrix[i, j] = 2 * thetas[i, j] - (thetas[i, j - 1] + thetas[i, j + 1])
            elif j == 0:
                objective_gradient_matrix[i, j] = 2 * thetas[i, j] - (thetas[i, s - 1] + thetas[i, 1])
            elif j == s - 1:
                objective_gradient_matrix[i, j] = 2 * thetas[i, j] - (thetas[i, s - 2] + thetas[i, 0])
    return objective_gradient_matrix


def constraint_gradient(thetas, lengths, constraint_number):
    assert isinstance(thetas, np.ndarray)
    n, s = thetas.shape
    col_index = None
    assert n == len(lengths)
    constraint_gradient_matrix = np.zeros((n, s))
    if constraint_number % 2 == 1:
        col_index = (constraint_number + 1) // 2 - 1
    elif constraint_number % 2 == 0:
        col_index = constraint_number // 2 - 1
    theta_cum_sum = np.cumsum(thetas[:, col_index])
    for i in range(n):
        elements_in_outer_sum = np.zeros(n)
        for j in range(n):
            if constraint_number % 2 == 1:
                elements_in_outer_sum[j] = - lengths[j] * np.sin(theta_cum_sum[j])
            elif constraint_number % 2 == 0:
                elements_in_outer_sum[j] = lengths[j] * np.cos(theta_cum_sum[j])
        constraint_gradient_matrix[i, col_index] = np.sum(elements_in_outer_sum[i:])
    return constraint_gradient_matrix


def constraint_squared_gradient(thetas, lengths, constraint_number):
    n, s = thetas.shape
    col_index = None
    assert n == len(lengths)
    constraint_gradient_matrix = np.zeros((n, s))
    if constraint_number % 2 == 1:
        col_index = (constraint_number + 1) // 2 - 1
    elif constraint_number % 2 == 0:
        col_index = constraint_number // 2 - 1
    theta_cum_sum = np.cumsum(thetas[:, col_index])
    final_sum = np.zeros(n)
    for row_index in range(n):
        partial_theta_sum = np.sum(thetas[row_index:, col_index])
        for i in range(n):
            for j in range(n):
                l_prod = lengths[i] * lengths[j]
                if constraint_number % 2 == 1:
                    if i < row_index <= j:
                        final_sum[row_index] += - l_prod * np.cos(theta_cum_sum[i]) * np.sin(partial_theta_sum)
                    elif j < row_index <= i:
                        final_sum[row_index] += - l_prod * np.sin(partial_theta_sum) * np.cos(theta_cum_sum[j])
                    elif i >= row_index and j >= row_index:
                        final_sum[row_index] += - l_prod * (np.sin(theta_cum_sum[i]) * np.cos(theta_cum_sum[j]) +
                                                            np.cos(theta_cum_sum[i]) * np.sin(theta_cum_sum[j]))
                elif constraint_number % 2 == 0:
                    if i < row_index <= j:
                        final_sum[row_index] += l_prod * np.sin(theta_cum_sum[i]) * np.cos(partial_theta_sum)
                    elif j < row_index <= i:
                        final_sum[row_index] += l_prod * np.cos(partial_theta_sum) * np.sin(theta_cum_sum[j])
                    elif i >= row_index and j >= row_index:
                        final_sum[row_index] += l_prod * (np.cos(theta_cum_sum[i]) * np.sin(theta_cum_sum[j]) +
                                                          np.sin(theta_cum_sum[i]) * np.cos(theta_cum_sum[j]))
    constraint_gradient_matrix[:, col_index] = final_sum
    return constraint_gradient_matrix


def constraint_grad_set(thetas, lengths):
    n, s = thetas.shape
    constraint_set = np.zeros((n * s, 2 * s))
    for i in range(1, 2 * s + 1):
        constraint_grad = constraint_gradient(thetas, lengths, i).flatten('F').reshape((n * s,))
        constraint_set[:, i - 1] = constraint_grad
    return constraint_set


def get_constraint_set(thetas, lengths, coordinates):
    n, s = thetas.shape
    constraint_set = np.zeros((2, s))
    for i in range(1, 2 * s + 1):
        if i % 2 == 0:
            col_index = i//2 - 1
            constraint_set[1, col_index] = constraint(thetas, lengths, i, coordinates[1, col_index])
        elif i % 2 == 1:
            col_index = (i+1)//2 - 1
            constraint_set[0, col_index] = constraint(thetas, lengths, i, coordinates[0, col_index])
    return constraint_set

def generate_quadratically_penealized_objective(robot_arm):
    '''
    Given a RobotArm object with a valid joint lenghts and destinations
    this function returns a quadratically penalized objective function
    taking in two parameters: theta and mu
    '''
    constraints = functools.partial(
        get_constraint_set,
        lengths=robot_arm.lengths,
        coordinates=robot_arm.destinations
    )
    def quadratically_penealized_objective(thetas, mu):
        return objective(thetas) + 0.5 * mu * np.sum(constraints**2)

    return quadratically_penealized_objective
