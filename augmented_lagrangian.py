import numpy as np

from methods import BFGS
from plotting import path_figure
from problem import objective, objective_gradient


def augmented_lagrangian_method(initial_penalty, initial_tolerance, initial_thetas,
                                initial_lambdas, max_iter, global_tolerance, constraint, constraint_grad, robot):
    n, s = initial_thetas.shape
    current_lambdas = initial_lambdas
    current_tolerance = initial_tolerance
    current_penalty = initial_penalty
    current_thetas = initial_thetas.flatten('F')
    for current_iterate in range(max_iter):
        l_c_sum = 0
        c_sq_sum = 0
        c_sq_sum_alternative = np.zeros(n * s)
        for i in range(1, 2 * s + 1):
            current_constraint = constraint(current_thetas)[i - 1]
            current_constraint_grad = constraint_grad(current_thetas)[:, i - 1]
            l_c_sum += current_lambdas[i - 1] * current_constraint
            c_sq_sum += current_constraint ** 2
            c_sq_sum_alternative += (current_lambdas[
                                         i - 1] - current_penalty * current_constraint) * current_constraint_grad

        def augmented_lagrangian(func1_thetas):
            return objective(
                func1_thetas.reshape((n, s), order='F')).flatten('F') - l_c_sum + current_penalty / 2 * c_sq_sum

        def grad_augmented_lagrangian(func2_thetas):
            grad_aug_lag = objective_gradient(func2_thetas.reshape((n, s), order='F')).flatten('F') \
                           - c_sq_sum_alternative
            return grad_aug_lag

        previous_thetas = current_thetas
        current_thetas = BFGS(current_thetas, augmented_lagrangian,
                              grad_augmented_lagrangian, current_tolerance)

        if np.linalg.norm(previous_thetas - current_thetas) == global_tolerance:
            path_figure(current_thetas.reshape((n, s), order='F'), robot)
            return current_thetas

        for j in range(2 * s):
            current_lambdas[j] -= current_penalty * constraint(current_thetas)[j]

        current_penalty *= 2

        current_tolerance /= 1

    return None

# def quadratic(func1_thetas):
#   return objective(func1_thetas.reshape((n, s), order='F')).flatten('F') + current_penalty / 2 * c_sq_sum
# def quadratic_grad(func2_thetas):
#   return objective_gradient(func2_thetas.reshape((n, s), order='F')).flatten('F') + current_penalty * grad_c_sq_sum
# grad_c_sq_sum = 0
# grad_c_sq_sum += current_constraint * current_constraint_grad
