import numpy as np


def generate_constraints_function(robot_arm):
    n = robot_arm.n
    s = robot_arm.s

    def constraints(thetas):
        '''
        Given a set of s n-dimensional theta vectors column-wise in an array,
        returns the value of 2s equality constrains in a 1D numpy array
        '''
        # The function must map from R^(ns) --> R^(2s)
        if not thetas.shape == (n * s,):
            raise ValueError('Thetas is not given as 1D vector')

        # Calculate the positions that the robot will take given the thetas
        positions = np.apply_along_axis(
            robot_arm.position,
            axis=0,
            arr=thetas.reshape((n, s), order='F')
        )
        assert positions.shape == (2, s)

        return (positions - robot_arm.destinations).reshape((2 * s,), order='F')

    return constraints


def generate_constraint_gradients_function(robot_arm):
    n = robot_arm.n
    s = robot_arm.s

    def constraint_gradients(thetas):
        # The function must map from R^(ns) --> R^(ns x 2s)
        if not thetas.shape == (n * s,):
            raise ValueError('Thetas is not given as 1D vector')

        joint_angles = np.cumsum(thetas.reshape((n, s)), axis=0)
        assert joint_angles.shape == (n, s,)
        x_components = robot_arm.lengths.reshape(n, 1) * np.cos(joint_angles)
        y_components = robot_arm.lengths.reshape(n, 1) * np.sin(joint_angles)

        def a(q, j):
            return np.sum(x_components[q:, j], dtype=np.float)

        def b(q, j):
            return np.sum(y_components[q:, j], dtype=np.float)

        gradients_matrix = np.zeros((n * s, 2 * s))

        for j in range(0, s):  # Constraint number
            for q in range(0, n):  # Joint angle number
                gradients_matrix[j * n + q, 2 * j] = -b(q, j)
                gradients_matrix[j * n + q, 2 * j + 1] = a(q, j)

        return gradients_matrix

    return constraint_gradients
