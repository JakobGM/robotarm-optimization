import unittest
import numpy as np

from robot_arm import RobotArm
from quadratic_penalty import (
    generate_quadratically_penalized_objective,
    generate_quadratically_penalized_objective_gradient,
    quadratic_penalty_method,
)
from plotting import path_figure


class TestGeneratorFunctions(unittest.TestCase):
    def setUp(self):
        lengths = (3, 2, 2,)
        destinations = (
            (5, 4, 6, 4, 5),
            (0, 2, 0.5, -2, -1),
        )
        theta = (np.pi, np.pi / 2, 0,)
        self.robot_arm = RobotArm(lengths, destinations, theta)
        self.n = self.robot_arm.n
        self.s = self.robot_arm.s

    def test_penalized_objective_generator_invocation(self):
        generate_quadratically_penalized_objective(self.robot_arm)

    def test_penalized_objective_gradient_generator_invocation(self):
        generate_quadratically_penalized_objective_gradient(self.robot_arm)

    def test_quadratic_penalty_method(self):
        thetas = quadratic_penalty_method(self.robot_arm)
        path_figure(thetas.reshape((self.n, self.s,), order='F'), self.robot_arm, show=False)
