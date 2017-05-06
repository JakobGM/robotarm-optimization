import numpy as np

from constraints import generate_constraints_function, generate_constraint_gradients_function
from methods import BFGS
from plotting import path_figure
from problem import generate_objective_gradient_function, generate_objective_function
from robot_arm import RobotArm
import random


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
                                global_tolerance, max_iter, robot, generate_initial_guess=True):
    lagrange_multiplier_function = generate_lagrange_multiplier(robot)
    if generate_initial_guess is True:
        thetas = robot.generate_initial_guess()
    elif generate_initial_guess == "random":
        thetas = np.random.rand(robot.n*robot.s) * 2*np.pi
    else:
        thetas = np.zeros(robot.n * robot.s)
    mu = initial_penalty
    tolerance = initial_tolerance
    lagrange_multiplier = initial_lagrange_multiplier
    print("Starting augmented lagrangian method")
    for i in range(max_iter):
        augmented_lagrangian_objective = generate_augmented_lagrangian_objective(robot, lagrange_multiplier, mu)
        augmented_lagrangian_objective_gradient = generate_augmented_lagrangian_objective_gradient(robot,
                                                                                                   lagrange_multiplier,
                                                                                                   mu)
        thetas = BFGS(thetas, augmented_lagrangian_objective, augmented_lagrangian_objective_gradient,
                      tolerance)
        lagrange_multiplier = lagrange_multiplier_function(thetas, lagrange_multiplier, mu)
        mu *= 1.4
        tolerance *= 0.7

        current_norm = np.linalg.norm(augmented_lagrangian_objective_gradient(thetas))
        if current_norm < global_tolerance:
            path_figure(thetas.reshape((robot.n, robot.s), order='F'), robot)
            print("Augmented lagrangian method successful")
            return thetas

    print("Augmented lagrangian method unsuccessful")


coordinates_tuple = ((5, 3, -7, 9, 4), (4, 2, 3, 2, 1))
lengths_tuple = (2, 3, 4, 2)
robot = RobotArm(lengths_tuple, coordinates_tuple)
lambdas = np.zeros(2 * robot.s)

augmented_lagrangian_method(lambdas, 4, 1e-2, 5e-3, 100, robot)
