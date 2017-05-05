import unittest
import numpy as np

from robot_arm import RobotArm
from quadratic_penalty import (
    generate_quadratically_penalized_objective,
    generate_quadratically_penalized_objective_gradient,
)


class TestGeneratorFunctions(unittest.TestCase):
    def setUp(self):
        lengths = (3, 2, 2,)
        destinations = (
            (5, 4, 6, 4, 5),
            (0, 2, 0.5, -2, -1),
        )
        theta = (np.pi, np.pi / 2, 0,)
        self.robot_arm = RobotArm(lengths, destinations, theta)

    def test_penalized_objective_generator_invocation(self):
        generate_quadratically_penalized_objective(self.robot_arm)

    def test_penalized_objective_gradient_generator_invocation(self):
        generate_quadratically_penalized_objective_gradient(self.robot_arm)
