from problem import generate_objective_gradient_function, generate_objective_function
import numpy as np


def generate_extended_objective_gradient_function(robot_arm, barrier_penalty):
    objective_gradient_function = generate_objective_gradient_function(robot_arm)
    n = robot_arm.n
    s = robot_arm.s

    def extended_objective_gradient_function(thetas_slack):
        thetas = thetas_slack[:n*s]
        slack = thetas_slack[n*s:]
        assert thetas_slack.shape == (n*s + 2*n*s, )
        objective_gradient = objective_gradient_function(thetas)
        slack_objective_gradient = 1/slack * barrier_penalty
        return np.append(objective_gradient, slack_objective_gradient)
    return extended_objective_gradient_function


def generate_extended_objective_function(robot_arm, barrier_penalty):
    objective_function = generate_objective_function(robot_arm)
    n = robot_arm.n
    s = robot_arm.s

    def extended_objective(thetas_slack):
        thetas = thetas_slack[:n*s]
        slack = thetas_slack[n*s:]
        return objective_function(thetas) - barrier_penalty*np.sum(np.log(slack))
    return extended_objective
