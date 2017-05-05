import unittest
import numbers
import numpy as np
from problem import (
    generate_objective_function,
    generate_quadratically_penalized_objective,
    generate_quadratically_penalized_objective_gradient,
)
from robot_arm import RobotArm

class TestObjectiveFunction(unittest.TestCase):
    def setUp(self):
        self.thetas = np.array((0.5, 4, 2, 5, 3, 6,))
        self.thetas.setflags(write=False)
        lengths = (1, 1,)
        destinations = ((0, 0, 0), (0, 0, 0))
        self.robot_arm = RobotArm(lengths, destinations)
        self.objective = generate_objective_function(self.robot_arm)

    def test_theta_size(self):
        self.assertEqual(self.thetas.shape, (2*3,))

    def test_return_type(self):
        self.assertIsInstance(self.objective(self.thetas), numbers.Number)

    def test_correct_objective_value(self):
        self.assertEqual(self.objective(self.thetas), 31 / 4)

    def test_immutability(self):
        original = self.thetas
        self.objective(self.thetas)
        np.testing.assert_equal(original, self.thetas)


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
