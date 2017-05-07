import numpy as np

from extended_constraints import generate_extended_constraints_gradient_function, generate_extended_constraints_function
from extended_objective import generate_extended_objective_function, generate_extended_objective_gradient_function
from methods import BFGS
from plotting import path_figure
from robot_arm import RobotArm


def generate_extended_augmented_objective(robot, barrier_penalty, lagrange_multiplier, mu):
    extended_objective = generate_extended_objective_function(robot, barrier_penalty)
    extended_constraint = generate_extended_constraints_function(robot)

    def extended_augmented_objective(thetas_slack):
        extended_objective(thetas_slack) - np.sum(lagrange_multiplier(thetas_slack) * extended_constraint) - mu / 2 * np.sum(
            extended_constraint ** 2)

    return extended_augmented_objective


def generate_extended_augmented_gradient(robot, barrier_penalty, lagrange_multiplier, mu):
    n = robot.n
    s = robot.s
    extended_objective_gradient = generate_extended_objective_gradient_function(robot, barrier_penalty)
    extended_constraint = generate_extended_constraints_function(robot)
    extended_constraint_gradient = generate_extended_constraints_gradient_function(robot)

    def extended_augmented_gradient(thetas_slack):
        second_part = 0
        for i in range(2 * s):
            second_part += (lagrange_multiplier[i] - mu * extended_constraint(thetas_slack)[i]) * extended_constraint_gradient(thetas_slack)[:, i]

        return extended_objective_gradient(thetas_slack) - second_part

    return extended_augmented_gradient


def generate_extended_lagrange_multiplier(robot_arm):
    extended_constraints_function = generate_extended_constraints_function(robot_arm)

    def lagrange_multiplier(thetas_slack, old_lagrange_multiplier, mu):
        return old_lagrange_multiplier - mu * extended_constraints_function(thetas_slack)

    return lagrange_multiplier


def extended_augmented_lagrangian_method(initial_lagrange_multiplier, initial_penalty, initial_tolerance,
                                         global_tolerance, max_iter, robot, barrier_penalty,
                                         generate_initial_guess=False,
                                         convergence_analysis=False):

    lagrange_multiplier_function = generate_extended_lagrange_multiplier(robot)

    if generate_initial_guess is True:
        thetas = robot.generate_initial_guess()
    elif generate_initial_guess == "random":
        thetas = np.random.rand(robot.n * robot.s) * 2 * np.pi
    else:
        thetas = np.zeros(robot.n*robot.s + 2*robot.n*robot.s, )

    mu = initial_penalty
    tolerance = initial_tolerance
    lagrange_multiplier = initial_lagrange_multiplier

    # Saving the iterates if convergence analysis is active
    if convergence_analysis is True:
        iterates = thetas.copy().reshape((robot.n * robot.s, 1))

    print("Starting augmented lagrangian method")
    for i in range(max_iter):
        augmented_lagrangian_objective = generate_extended_augmented_objective(robot, barrier_penalty,
                                                                               lagrange_multiplier, mu)
        augmented_lagrangian_objective_gradient = generate_extended_augmented_gradient(robot, barrier_penalty,
                                                                                       lagrange_multiplier,
                                                                                       mu)
        thetas = BFGS(thetas, augmented_lagrangian_objective, augmented_lagrangian_objective_gradient,
                      tolerance)
        lagrange_multiplier = lagrange_multiplier_function(thetas, lagrange_multiplier, mu)
        mu *= 1.4
        tolerance *= 0.7

        # Adding the new thetas to iterates used for convergence analysis
        if convergence_analysis is True:
            iterates = np.concatenate((iterates, thetas.reshape(robot.n * robot.s, 1)), axis=1)

        current_norm = np.linalg.norm(augmented_lagrangian_objective_gradient(thetas))
        if current_norm < global_tolerance:
            path_figure(thetas.reshape((robot.n, robot.s), order='F'), robot, show=False)

            print("Augmented lagrangian method successful")
            if convergence_analysis is True:
                return iterates, generate_objective_function(robot)
            else:
                return thetas

    print("Augmented lagrangian method unsuccessful")


coordinates_tuple = ((5, 4, 6, 4, 5), (0, 2, 0.5, -2, -1))
lengths_tuple = (3, 2, 2)
robot = RobotArm(lengths_tuple, coordinates_tuple, angular_constraint=np.pi / 2)
lagrange_multipliers = np.zeros(2 * robot.s * (robot.n + 1))
extended_augmented_lagrangian_method(lagrange_multipliers, 10, 1e-2, 1e-2, 100, robot, 10,
                                     generate_initial_guess=False, convergence_analysis=False)
