import unittest
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from robot_arm import RobotArm
from plotting import path_figure

class TestPlotting(unittest.TestCase):
    def setUp(self):
        lengths = (3, 2, 2,)
        destinations = (
            (5, 4, 6, 4, 5),
            (0, 2, 0.5, -2, -1),
        )
        theta = (pi, pi / 2, 0,)
        self.robot_arm = RobotArm(
            lengths=lengths,
            destinations=destinations,
            theta=theta
        )
        n = len(lengths)
        s = len(destinations[0])
        total_joints = n * s
        self.theta_matrix = np.arange(total_joints).reshape((n, s))

    def test_plot_pure_functon(self):
        # Save values before function invocation
        original_destinations = self.robot_arm.destinations.copy()
        original_theta_matrix = self.theta_matrix.copy()

        # Run the pure function
        path_figure(self.theta_matrix, self.robot_arm, show=False)

        # Assert that none of the arguments have been changed
        np.testing.assert_array_equal(original_destinations, self.robot_arm.destinations)
        np.testing.assert_array_equal(original_theta_matrix, self.theta_matrix)
