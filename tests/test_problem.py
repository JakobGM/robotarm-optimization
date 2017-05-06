import unittest
from numpy import testing
import numbers
import numpy as np
from problem import generate_objective_function, generate_objective_gradient_function
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
        original = self.thetas.copy()
        self.objective(self.thetas)
        np.testing.assert_equal(original, self.thetas)

    def test_objective_value2(self):
        robot_arm = RobotArm(
            lengths=(1, 2,),
            destinations=((1, -1,), (1, 1,),)
        )
        thetas = np.array((0, np.pi/2, np.pi/2, np.pi/4,))

        objective_func = generate_objective_function(robot_arm)
        objective_value = objective_func(thetas)
        self.assertEqual(objective_value, np.pi**2 * 5/16)

        objective_gradient_func = generate_objective_gradient_function(robot_arm)
        objective_gradient = objective_gradient_func(thetas)
        testing.assert_array_equal(objective_gradient, np.array((-np.pi, np.pi/2, np.pi, -np.pi/2,)))
