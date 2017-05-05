import unittest

import numpy as np

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
