import numpy as np

from problem import objective, objective_gradient, constraint, constraint_gradient, constraint_squared_gradient, \
    get_constraint_set
from methods import BFGS


def augmented_lagrangian_method(initial_penalty, initial_tolerance, initial_thetas, initial_lambdas, lengths,
                                coordinates, max_iter, global_tolerance):
    n, s = initial_thetas.shape
    for current_iterate in range(max_iter):
        current_tolerance = initial_tolerance
        current_thetas = initial_thetas
        current_lambdas = initial_lambdas
        current_penalty = initial_penalty
        l_c_sum = 0
        c_sq_sum = 0
        l_grad_c_sum = 0
        grad_c_sq_sum = 0
        for i in range(1, 2 * s + 1):
            current_constraint = constraint(current_thetas, lengths, i, coordinates)
            current_constraint_grad = constraint_gradient(current_thetas, lengths, i)
            current_constraint_sq_grad = constraint_squared_gradient(current_thetas, lengths, i)
            l_c_sum += current_lambdas[i - 1] * current_constraint
            c_sq_sum += current_constraint ** 2
            l_grad_c_sum += current_lambdas[i - 1] * current_constraint_grad
            grad_c_sq_sum += current_constraint_sq_grad

        def augmented_lagrangian(func1_thetas):
            return objective(func1_thetas) - l_c_sum + current_penalty / 2 * c_sq_sum

        def grad_augmented_lagrangian(func2_thetas):
            return objective_gradient(func2_thetas) - l_grad_c_sum + current_penalty / 2 * grad_c_sq_sum

        current_thetas = BFGS(current_thetas, augmented_lagrangian, grad_augmented_lagrangian, current_tolerance)

        if np.linalg.norm(get_constraint_set(current_thetas, lengths, coordinates)) < global_tolerance:
            return current_thetas

        for p in range(1, 2 * s + 1):
            current_lambdas[p - 1] -= current_penalty * constraint(current_thetas, lengths, p,
                                                                   coordinates.flatten('F')[p - 1])

        current_penalty **= 2

        current_tolerance /= 2

    return None
