import unittest
import numbers
import numpy as np
from problem import (objective, objective_gradient, constraint, constraint_gradient,
                     constraint_squared_gradient, constraint_grad_set)


class TestObjectiveFunction(unittest.TestCase):
    def setUp(self):
        self.thetas = np.array(((0.5, 2, 3,), (4, 5, 6,),))
        self.thetas.setflags(write=False)

    def test_theta_size(self):
        self.assertEqual(self.thetas.shape, (2, 3))

    def test_return_type(self):
        self.assertIsInstance(objective(self.thetas), numbers.Number)

    def test_correct_objective_value(self):
        self.assertEqual(objective(self.thetas), 31 / 4)

    def test_immutability(self):
        original = self.thetas
        objective(self.thetas)
        np.testing.assert_equal(original, self.thetas)


class TestConstraintFunction(unittest.TestCase):
    def setUp(self):
        self.thetas = np.array(((0.5, 2, 3,), (4, 5, 6,),))
        self.lengths = np.array((3, 2,))
        self.n, self.s = self.thetas.shape
        self.constraint_set = np.arange(1, 2 * self.s + 1, dtype=int)
        self.coordinates = np.array(((3, 1, 4), (1, 2, 3)))

    def test_theta_size(self):
        self.assertEqual(self.thetas.shape, (2, 3))

    def test_lengths_size(self):
        self.assertEqual(self.lengths.shape, (2,))

    def test_constraint_set_size(self):
        self.assertEqual(self.constraint_set.shape, (2 * self.s,))

    def test_return_type(self):
        self.assertIsInstance(constraint(self.thetas, self.lengths, self.constraint_set[4], self.coordinates[0, 2]),
                              numbers.Number)

    def test_correct_constraint_value(self):
        correct_value = 3 * np.cos(3) + 2 * np.cos(3 + 6) - 4
        self.assertEqual(constraint(self.thetas, self.lengths, self.constraint_set[4], self.coordinates[0, 2]),
                         correct_value)


class TestObjectiveGradientFunction(unittest.TestCase):
    def setUp(self):
        self.thetas = np.array(((0.5, 2, 3,), (4, 5, 6,),))
        self.thetas.setflags(write=False)

    def test_theta_size(self):
        self.assertEqual(self.thetas.shape, (2, 3))

    def test_return_size(self):
        self.assertEqual(objective_gradient(self.thetas).shape, (2, 3))

    def test_correct_objective_gradient_value(self):
        objective_gradient_matrix = np.array(((-4, 1 / 2, 7 / 2,), (-3, 0, 3,),))
        np.testing.assert_equal(objective_gradient(self.thetas), objective_gradient_matrix)


class TestConstraintGradientFunction(unittest.TestCase):
    def setUp(self):
        self.thetas = np.array(((0.5, 2, 3,), (4, 5, 6,),))
        self.lengths = np.array((3, 2,))
        self.n, self.s = self.thetas.shape
        self.constraint_set = np.arange(1, 2 * self.s + 1, dtype=int)

    def test_n_and_s(self):
        self.assertEqual(self.n, self.lengths.shape[0])
        self.assertEqual(self.s, self.thetas.shape[1])

    def test_theta_size(self):
        self.assertEqual(self.thetas.shape, (2, 3))

    def test_lengths_size(self):
        self.assertEqual(self.lengths.shape, (2,))

    def test_constraint_set_size(self):
        self.assertEqual(self.constraint_set.shape, (2 * self.s,))

    def test_return_size(self):
        self.assertEqual(constraint_gradient(self.thetas, self.lengths, self.constraint_set[4]).shape, (2, 3))

    def test_correct_constraint_gradient_value(self):
        correct_value = -(3 * np.sin(3) + 2 * np.sin(3 + 6))
        self.assertEqual(constraint_gradient(self.thetas, self.lengths, self.constraint_set[4])[0, 2], correct_value)


class TestConstraintSquaredGradientFunction(unittest.TestCase):
    def setUp(self):
        self.thetas = np.array(((0.5, 2, 3,), (4, 5, 6,),))
        self.lengths = np.array((3, 2,))
        self.n, self.s = self.thetas.shape
        self.constraint_set = np.arange(1, 2 * self.s + 1, dtype=int)

    def test_n_and_s(self):
        self.assertEqual(self.n, self.lengths.shape[0])
        self.assertEqual(self.s, self.thetas.shape[1])

    def test_theta_size(self):
        self.assertEqual(self.thetas.shape, (2, 3))

    def test_lengths_size(self):
        self.assertEqual(self.lengths.shape, (2,))

    def test_constraint_set_size(self):
        self.assertEqual(self.constraint_set.shape, (2 * self.s,))

    def test_return_size(self):
        self.assertEqual(constraint_squared_gradient(self.thetas, self.lengths, self.constraint_set[4]).shape,
                         (2, 3))

    def test_correct_constraint_squared_gradient_value(self):
        correct_value = -(
            3 * 3 * (2 * np.sin(3) * np.cos(3)) + 3 * 2 * (np.sin(3) * np.cos(3 + 6) + np.cos(3) * np.sin(3 + 6)) +
            2 * 3 * (np.sin(3 + 6) * np.cos(3) + np.cos(3 + 6) * np.sin(3)) + 2 * 2 * (
                2 * np.sin(3 + 6) * np.cos(3 + 6)))
        self.assertEqual(constraint_squared_gradient(self.thetas, self.lengths, self.constraint_set[4])[0, 2],
                         correct_value)


class TestConstraintGradientSet(unittest.TestCase):
    def setUp(self):
        self.thetas = np.array(((0.5, 2, 3,), (4, 5, 6,),))
        self.lengths = np.array((3, 2,))

    def test_set_size(self):
        self.assertEqual(constraint_grad_set(self.thetas, self.lengths).shape, (6, 6))
