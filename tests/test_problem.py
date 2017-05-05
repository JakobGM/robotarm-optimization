import unittest
import numbers
import numpy as np
from problem import generate_objective_function
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
