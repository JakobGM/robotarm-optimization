import unittest

from convergence_analysis import plot_convergence
from fixtures import robot_arm1
from augmented_lagrangian import generate_simple_augmented_lagrangian_function


class TestConvergenceAnalysis(unittest.TestCase):

    def setUp(self):
        self.augmented_lagrangian_method = generate_simple_augmented_lagrangian_function()
        self.thetas_matrix, self.objective_func = self.augmented_lagrangian_method(robot_arm1)

    def test_convergence_plot_invocation(self):
        print(self.thetas_matrix)
        plot_convergence(self.thetas_matrix, self.objective_func, robot_arm1, show=True)
