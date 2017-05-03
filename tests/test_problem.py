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
        self.constraint_set = np.arange(1, 13, dtype=int)
        self.coordinates = np.array((3, 1, 4), (1, 2, 3))

    def test_constraint_value(self):
        self.assertEqual(constraint(self.thetas, self.lengths, self.constraint_set[0]),)



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
        self.constraint_set = np.arange(1, 13, dtype=int)

    def test_theta_size(self):
        self.assertEqual(self.thetas.shape, (2, 3))

    def test_lengths_size(self):
        self.assertEqual(len(self.lengths), 2)

    def test_constraint_set_size(self):
        self.assertEqual(len(self.constraint_set), 12)

    def test_return_size(self):
        self.assertEqual(objective_gradient(self.thetas).shape, (2, 3))


class TestConstraintSquaredGradientFunction():
    pass

class TestConstraintGradientSet():
    pass
