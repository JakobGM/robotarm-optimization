import unittest
import numpy as np
from numpy import pi

from robot_arm import RobotArm


class TestRobotArm(unittest.TestCase):
    def setUp(self):
        self.lengths = (3, 2, 2,)
        self.destinations = (
            (5, 4, 6, 4, 5),
            (0, 2, 0.5, -2, -1),
        )
        self.theta = (pi, pi / 2, 0,)
        self.robot_arm = RobotArm(self.lengths, self.destinations, self.theta)

    def test_init_all_arguments(self):
        RobotArm(self.lengths, self.destinations, self.theta)

    def test_init_without_theta(self):
        RobotArm(self.lengths, self.destinations)

    def test_wrong_lengths_type(self):
        self.assertRaises(
            TypeError,
            RobotArm,
            np.array(self.lengths),
            self.destinations,
            self.theta
        )

    def test_wrong_destinations_type(self):
        self.assertRaises(
            TypeError,
            RobotArm,
            self.lengths,
            np.array(self.destinations),
            self.theta)

    def test_wrong_theta_type(self):
        self.assertRaises(
            TypeError,
            RobotArm,
            self.lengths,
            self.destinations,
            np.array(self.theta))

    def test_destinations_properties(self):
        robot_arm = RobotArm(self.lengths, self.destinations, self.theta)
        self.assertIsInstance(robot_arm.destinations, np.ndarray)

        # Check if points are 2D
        self.assertTrue(robot_arm.destinations.shape[0] == 2)

        # Check if destinations are immutable
        self.assertRaises(
            ValueError,
            robot_arm.destinations.__setitem__,
            (0, 0,),
            0,
        )

    def test_generate_initial_guess(self):
        self.robot_arm.generate_initial_guess()
