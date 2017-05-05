import numpy as np
import unittest

from constraints import generate_constraints_function
from robot_arm import RobotArm

class TestConstraintFunction(unittest.TestCase):
    def setUp(self):
        self.lengths = (3, 2, 2,)
        self.destinations = (
            (5, 4, 6, 4, 5),
            (0, 2, 0.5, -2, -1),
        )
        self.theta = (np.pi, np.pi / 2, 0,)
        self.thetas = np.ones((3 * 5,))
        self.robot_arm = RobotArm(self.lengths, self.destinations, self.theta)
        self.constraint_func = generate_constraints_function(self.robot_arm)

    def test_constraints_func_return_type(self):
        constraints = self.constraint_func(self.thetas)
        self.assertEquals(constraints.shape, (2*5,))
