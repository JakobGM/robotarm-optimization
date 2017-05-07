import unittest
import numpy as np

from plotting import path_figure
from fixtures import robot_arm1

class TestPlotting(unittest.TestCase):
    def setUp(self):
        self.robot_arm = robot_arm1
        n = len(self.robot_arm.lengths)
        s = len(self.robot_arm.destinations[0])
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
