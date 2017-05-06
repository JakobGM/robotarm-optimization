import numpy as np

from constraints import generate_constraints_function, generate_constraint_gradients_function
from methods import BFGS
from plotting import path_figure
from problem import generate_objective_gradient_function, generate_objective_function
from robot_arm import RobotArm


def generate_lagrange_multiplier(robot_arm):
    constraints_function = generate_constraints_function(robot_arm)

    def lagrange_multiplier(thetas, old_lagrange_multiplier, mu):
        return old_lagrange_multiplier - mu * constraints_function(thetas)

    return lagrange_multiplier


def generate_augmented_lagrangian_objective(robot_arm, lagrange_multiplier, mu):
    objective_function = generate_objective_function(robot_arm)
    constraint_function = generate_constraints_function(robot_arm)

    def augmented_lagrangian_objective(thetas):
        objective = objective_function(thetas)
        constraints = constraint_function(thetas)
        return objective - np.sum(lagrange_multiplier * constraints) + mu / 2 * np.sum(constraints * constraints)

    return augmented_lagrangian_objective


def generate_augmented_lagrangian_objective_gradient(robot_arm, lagrange_multiplier, mu):
    n = robot_arm.n
    s = robot_arm.s

    constraints_function = generate_constraints_function(robot_arm)
    objective_gradient_function = generate_objective_gradient_function(robot_arm)
    constraint_gradient_function = generate_constraint_gradients_function(robot_arm)

    def augmented_lagrangian_objective_gradient(thetas):
        constraints = constraints_function(thetas)
        objective_gradient = objective_gradient_function(thetas)
        constraint_gradient = constraint_gradient_function(thetas)

        second_part = 0
        for i in range(2 * s):
            second_part += (lagrange_multiplier[i] - mu * constraints[i]) * constraint_gradient[:, i]

        return objective_gradient - second_part

    return augmented_lagrangian_objective_gradient


def augmented_lagrangian_method(initial_lagrange_multiplier, initial_penalty, initial_tolerance,
                                global_tolerance, max_iter, robot):
    print("Starting augmented lagrangian method")
    lagrange_multiplier_function = generate_lagrange_multiplier(robot)
    thetas = robot.generate_initial_guess()
    mu = initial_penalty
    tolerance = initial_tolerance
    lagrange_multiplier = initial_lagrange_multiplier
    for i in range(max_iter):
        augmented_lagrangian_objective = generate_augmented_lagrangian_objective(robot, lagrange_multiplier, mu)
        augmented_lagrangian_objective_gradient = generate_augmented_lagrangian_objective_gradient(robot,
                                                                                                   lagrange_multiplier,
                                                                                                   mu)
        thetas = BFGS(thetas, augmented_lagrangian_objective, augmented_lagrangian_objective_gradient,
                      tolerance)
        lagrange_multiplier = lagrange_multiplier_function(thetas, lagrange_multiplier, mu)
        mu *= 1.5
        tolerance *= 0.9
    path_figure(thetas.reshape((robot.n, robot.s), order='F'), robot)
    return thetas


coordinates_tuple = ((1, 1, -1), (-1, 2, 3))
lengths_tuple = (2, 3, 4, 2)
robot = RobotArm(lengths_tuple, coordinates_tuple)
lambdas = np.zeros(2 * robot.s)

augmented_lagrangian_method(lambdas, 1, 1e-2, 1, 10, robot)
