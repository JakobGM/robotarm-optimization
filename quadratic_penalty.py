import numpy as np

from problem import (
    generate_objective_function,
    generate_objective_gradient_function,
)
from constraints import (
    generate_constraints_function,
    generate_constraint_gradients_function,
)

def generate_quadratically_penalized_objective(robot_arm):
    '''
    Given a RobotArm object with a valid joint lenghts and destinations
    this function returns a quadratically penalized objective function
    taking in two parameters: thetas and mu
    Function: R^(ns) x R --> R
    '''
    n = robot_arm.n
    s = robot_arm.s
    objective = generate_objective_function(robot_arm)
    constraints_func = generate_constraints_function(robot_arm)

    def quadratically_penalized_objective(thetas, mu):
        if not thetas.shape == (n * s,):
            raise ValueError('Thetas is not given as 1D-vector, but as: ' + \
                             str(thetas.shape))

        return objective(thetas) + \
               0.5 * mu * np.sum(constraints_func(thetas) ** 2)


    return quadratically_penalized_objective


def generate_quadratically_penalized_objective_gradient(robot_arm):
    n = robot_arm.n
    s = robot_arm.s

    objective_gradient = generate_objective_gradient_function(robot_arm)

    constraints_func = generate_constraints_function(robot_arm)
    constraint_gradients_func = generate_constraint_gradients_function(robot_arm)

    def quadratic_constraint_gradients(thetas):
        if not thetas.shape == (n * s,):
            raise ValueError('Thetas is not given as 1D-vector, but as: ' + \
                             str(thetas.shape))

        constraint_gradients = constraint_gradients_func(thetas)
        constraints = constraints_func(thetas)
        return constraints.reshape(1, 2*s) * constraint_gradients


    def quadratically_penalized_objective_gradient(thetas, mu):
        if not thetas.shape == (n * s,):
            raise ValueError('Thetas is not given as 1D-vector, but as: ' + \
                             str(thetas.shape))

        grad = objective_gradient(thetas)
        constraint_gradients = 0.5 * mu * quadratic_constraint_gradients(thetas)
        return grad + 0.5 * mu * np.sum(constraint_gradients, axis=1)


    return quadratically_penalized_objective_gradient
