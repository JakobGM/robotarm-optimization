import unittest
import numbers
import numpy as np
from numpy import pi

from robot_arm import RobotArm
from robot_arm import objective
from plotting import path_figure, plot_position


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

    def test_destinations_properties(self):
        robot_arm = RobotArm(self.lengths, self.destinations, self.theta)
        self.assertIsInstance(robot_arm.destinations, np.ndarray)

        # Check if points are 2D
        self.assertTrue(robot_arm.destinations.shape[0] == 2)

        # Check if destinations are immutable
        self.assertRaises(
            RuntimeError,
            robot_arm.destinations.__set_item__,
            0,
            0,
            None
        )


class TestObjectiveFunction(unittest.TestCase):

    def setUp(self):
        self.thetas = np.array(((0.5, 2, 3,), (4, 5, 6,),))
        self.thetas.setflags(write=False)

    def test_theta_size(self):
        self.assertEqual(self.thetas.shape, (2, 3))

    def test_return_type(self):
        self.assertIsInstance(objective(self.thetas), numbers.Number)

    def test_correct_objective_value(self):
        self.assertEqual(objective(self.thetas), 31/4)

    def test_immutability(self):
        original = self.thetas
        objective(self.thetas)
        np.testing.assert_equal(original, self.thetas)


class TestPlotting(unittest.TestCase):

    def setUp(self):
        lengths = (3, 2, 2,)
        destinations = (
                (5, 0,),
                (4, 2,),
                (6, 0.5),
                (4, -2),
                (5, -1),
        )
        theta = (pi, pi/2, 0,)
        self.robot_arm = RobotArm(
            lengths=lengths,
            destinations=destinations,
            theta=theta
        )
        n = len(lengths)
        s = len(destinations)
        total_joints = n*s
        self.theta_matrix = np.arange(total_joints).reshape((n, s))

    def test_path_figure_return(self):
        return_value = path_figure(self.theta_matrix, self.robot_arm)
        self.assertEqual(return_value, None)

    def test_plot_pure_functon(self):
        # Save values before function invocation
        original_destinations = self.robot_arm.destinations.copy()
        original_theta_matrix = self.theta_matrix.copy()

        # Run the pure function
        path_figure(self.theta_matrix, self.robot_arm)

        # Assert that none of the arguments have been changed
        np.testing.assert_array_equal(original_destinations, self.robot_arm.destinations)
        np.testing.assert_array_equal(original_theta_matrix, self.theta_matrix)
