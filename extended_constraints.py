import numpy as np

from constraints import generate_constraints_function, generate_constraint_gradients_function
from problem import generate_objective_gradient_function, generate_objective_function


def generate_extended_constraints_function(robot_arm):
    constraints_function = generate_constraints_function(robot_arm)
    n = robot_arm.n
    s = robot_arm.s

    def extended_constraints_function(thetas_slack):
        thetas = thetas_slack[:n*s]
        slack = thetas_slack[n*s:]
        assert thetas_slack.shape == (n*s + 2*n*s, )
        constraints = constraints_function(thetas)
        additional_constraints = np.zeros(2*n*s)
        additional_constraints[0::2] = thetas - robot_arm.angular_constraint - slack[0::2]
        additional_constraints[1::2] = robot_arm.angular_constraint - thetas - slack[1::2]
        return np.append(constraints, additional_constraints)

    return extended_constraints_function


def generate_extended_constraints_gradient_function(robot_arm):
    constraints_gradient_function = generate_constraint_gradients_function(robot_arm)
    n = robot_arm.n
    s = robot_arm.s

    def extended_constraints_gradient_function(thetas_slack):
        thetas = thetas_slack[:n*s]
        constraints_gradient = constraints_gradient_function(thetas)
        assert constraints_gradient.shape == (n*s, 2*s)
        downward_extension = np.zeros((2*n*s, 2*s))
        constraints_gradient = np.concatenate((constraints_gradient, downward_extension))
        assert constraints_gradient.shape == (3*n*s, 2*s)
        additional_constraints_gradient = np.zeros((3*n*s, 2*n*s))
        odd_additional_constraints_gradient_upper = np.identity(n * s)
        even_additional_constraints_gradient_upper = -np.identity(n * s)
        additional_constraints_gradient[:n * s, 0::2] = odd_additional_constraints_gradient_upper
        additional_constraints_gradient[:n * s, 1::2] = even_additional_constraints_gradient_upper
        additional_constraints_gradient[n * s:, :] = -np.identity(2 * n * s)
        return np.concatenate((constraints_gradient, additional_constraints_gradient), axis=1)

    return extended_constraints_gradient_function














