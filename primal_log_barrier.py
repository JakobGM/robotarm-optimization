import numpy as np

from extended_constraints import generate_extended_constraints_gradient_function, generate_extended_constraints_function
from extended_objective import generate_extended_objective_function, generate_extended_objective_gradient_function
from problem import generate_objective_function
from methods import BFGS, gradient_descent
from plotting import path_figure
from robot_arm import RobotArm
from config_space_angular_constraints import is_close_to_config_space
from scipy.optimize import minimize


def generate_extended_augmented_objective(robot, barrier_penalty, lagrange_multiplier, mu):
    extended_objective = generate_extended_objective_function(robot, barrier_penalty)
    extended_constraint = generate_extended_constraints_function(robot)

    def extended_augmented_objective(thetas_slack):
        return extended_objective(thetas_slack) - np.sum(lagrange_multiplier * extended_constraint(thetas_slack)) - mu / 2 * np.sum(
            extended_constraint(thetas_slack) ** 2)

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
    for i in range(robot.s):
        if not is_close_to_config_space(robot.destinations[:, i], robot.neighbor_tree, robot.lengths):
            print("Destination points not in configuration space")
            return None

    lagrange_multiplier_function = generate_extended_lagrange_multiplier(robot)

    if generate_initial_guess is True:
        thetas_slack = np.zeros(3 * robot.n * robot.s, )
        thetas_slack[:robot.n * robot.s] = robot.generate_initial_guess() + np.pi
        thetas_slack[robot.n * robot.s::2] = thetas_slack[:robot.n * robot.s] + robot.angular_constraint
        thetas_slack[robot.n * robot.s + 1::2] = robot.angular_constraint - thetas_slack[:robot.n * robot.s]
    elif generate_initial_guess == "random":
        thetas_slack = np.zeros(3 * robot.n * robot.s, )
        thetas_slack[:robot.n * robot.s] = np.random.rand(robot.n * robot.s) * 2 * np.pi
        thetas_slack[robot.n * robot.s::2] = thetas_slack[:robot.n * robot.s] + robot.angular_constraint
        thetas_slack[robot.n * robot.s + 1::2] = robot.angular_constraint - thetas_slack[:robot.n * robot.s]

    else:
        thetas_slack = np.zeros(3*robot.n*robot.s, )

    mu = initial_penalty
    tolerance = initial_tolerance
    lagrange_multiplier = initial_lagrange_multiplier

    # Saving the iterates if convergence analysis is active
    if convergence_analysis is True:
        iterates = thetas_slack.copy().reshape((3*robot.n*robot.s, 1))

    print("Starting augmented lagrangian method")
    for i in range(max_iter):
        augmented_lagrangian_objective = generate_extended_augmented_objective(robot, barrier_penalty,
                                                                               lagrange_multiplier, mu)
        augmented_lagrangian_objective_gradient = generate_extended_augmented_gradient(robot, barrier_penalty,
                                                                                       lagrange_multiplier,
                                                                                       mu)
        previous_thetas_slack = thetas_slack
        thetas_slack = BFGS(thetas_slack, augmented_lagrangian_objective, augmented_lagrangian_objective_gradient,
                      tolerance)

        lagrange_multiplier = lagrange_multiplier_function(thetas_slack, lagrange_multiplier, mu)
        mu *= 5
        tolerance *= 0.9

        # Adding the new thetas to iterates used for convergence analysis
        if convergence_analysis is True:
            iterates = np.concatenate((iterates, thetas_slack.reshape(robot.n * robot.s, 1)), axis=1)

        current_norm = np.linalg.norm(augmented_lagrangian_objective_gradient(thetas_slack))
        if current_norm < global_tolerance or np.linalg.norm(previous_thetas_slack - thetas_slack) < 0.01:
            path_figure(thetas_slack[:robot.n*robot.s].reshape((robot.n, robot.s), order='F'), robot, show=True)

            print("Augmented lagrangian method successful")
            if convergence_analysis is True:
                return iterates, generate_extended_objective_function
            else:
                return thetas_slack

    path_figure(thetas_slack[:robot.n * robot.s].reshape((robot.n, robot.s), order='F'), robot, show=True)
    print("Augmented lagrangian method unsuccessful")


coordinates_tuple = ((5, 4, 6, 4, 5), (0, 2, 0.5, -2, -1))
lengths_tuple = (3, 1, 1, 1, 1)
robot = RobotArm(lengths_tuple, coordinates_tuple, angular_constraint=np.pi / 2)
lagrange_multipliers = np.zeros(2 * robot.s * (robot.n + 1))
extended_augmented_lagrangian_method(lagrange_multipliers, 0.1, 1e-2, 1e-3, 20, robot, 1e-6,
                                     generate_initial_guess=True, convergence_analysis=False)
