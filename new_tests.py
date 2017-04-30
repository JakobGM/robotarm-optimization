from numpy import testing
import unittest
import numpy as np
from numpy import pi

from robot_arm import RobotArm


class TestRobotArm(unittest.TestCase):

    def setUp(self):
        self.lengths = (3, 2, 2,)
        self.destinations = (
                (5, 0,),
                (4, 2,),
                (6, 0.5),
                (4, -2),
                (5, -1),
        )
        self.theta = (pi, pi/2, 0,)

    def test_init_all_arguments(self):
        RobotArm(self.lengths, self.destinations, self.theta)

    def test_init_without_theta(self):
        RobotArm(self.lengths, self.destinations)

    def test_wrong_lengths_type(self):
        self.assertRaises(
                AssertionError,
                RobotArm,
                np.array(self.lengths),
                self.destinations,
                self.theta)

    def test_wrong_destinations_type(self):
        self.assertRaises(
                AssertionError,
                RobotArm,
                self.lengths,
                np.array(self.destinations),
                self.theta)

    def test_wrong_theta_type(self):
        self.assertRaises(
                AssertionError,
                RobotArm,
                self.lengths,
                self.destinations,
                np.array(self.theta))
