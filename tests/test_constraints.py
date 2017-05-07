import unittest

import numpy as np
from numpy import testing

from constraints import (generate_constraints_function,
                         generate_constraint_gradients_function, )
from robot_arm import RobotArm


class TestConstraintFunctions(unittest.TestCase):
    def setUp(self):
        self.lengths = (3, 2, 2,)
        self.destinations = (
            (5, 4, 6, 4, 5),
            (0, 2, 0.5, -2, -1),
        )
        self.theta = (np.pi, np.pi / 2, 0,)
        self.thetas = np.ones((3 * 5,))
        self.robot_arm = RobotArm(self.lengths, self.destinations, self.theta)
        self.constraints_func = generate_constraints_function(self.robot_arm)
        self.constraint_gradients_func = generate_constraint_gradients_function(self.robot_arm)

    def test_constraints_func_return_type(self):
        constraints = self.constraints_func(self.thetas)
        self.assertEqual(constraints.shape, (2 * 5,))

    def test_constraint_gradients_func_return_type(self):
        constraint_gradients = self.constraint_gradients_func(self.thetas)
        self.assertEqual(constraint_gradients.shape, (3 * 5, 2 * 5))
        # print(np.array2string(constraint_gradients, max_line_width=np.inf))

    def test_licq(self):
        constraint_gradients = self.constraint_gradients_func(self.thetas)
        rank = np.linalg.matrix_rank(constraint_gradients)
        self.assertEqual(rank, 2 * 5)

    def test_constraint_values(self):
        initial_guess = self.robot_arm.generate_initial_guess()
        constraints = self.constraints_func(initial_guess)
        constraint_grads = self.constraint_gradients_func(initial_guess)
        print(constraints)
        print(np.array2string(constraint_grads, max_line_width=np.inf))

    def test_constraint_values2(self):
        robot_arm = RobotArm(
            lengths=(1, 2,),
            destinations=((1, -1,), (1, 1,),)
        )
        thetas = np.array((0, np.pi/2, np.pi/2, np.pi/4,))

        constraints_func = generate_constraints_function(robot_arm)
        constraint_values = constraints_func(thetas)
        correct_constraint_values = np.array((0, 1, 1 - np.sqrt(2), np.sqrt(2),))
        testing.assert_array_almost_equal(constraint_values, correct_constraint_values)

        constraint_gradients_func = generate_constraint_gradients_function(robot_arm)
        correct = (
            (-2, 1, 0, 0,),
            (-2, 0, 0, 0,),
            (0, 0, -1 - np.sqrt(2), -np.sqrt(2),),
            (0, 0, -np.sqrt(2), -np.sqrt(2),),
        )
        constraint_grads = constraint_gradients_func(thetas)
        testing.assert_array_almost_equal(constraint_grads, np.array(correct))
