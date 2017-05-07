import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ImageMagickWriter

import plotting
from methods import BFGS
from plotting import path_figure
from quadratic_penalty import generate_quadratically_penalized_objective


class RobotArm:
    def __init__(self, lengths, destinations, theta=None, precision=1e-2, angular_constraint=None):
        # Input validation
        if not len(destinations) == 2:
            raise ValueError('Destinations are not in R2')
        if not len(destinations[0]) == len(destinations[1]):
            raise ValueError('Not the same number of x- and y-coordinates')
        if not isinstance(lengths, tuple) or not isinstance(destinations, tuple):
            raise TypeError('Arguments should be given as tuples')

        # Object attributes
        self.lengths = np.array(lengths)
        self.lengths.setflags(write=False)
        self.reach = np.sum(lengths)
        self.n = len(lengths)
        self.destinations = np.array(destinations)
        self.destinations.setflags(write=False)
        self.s = self.destinations.shape[1]
        self.precision = precision
        if angular_constraint is not None:
            self.angular_constraint = angular_constraint

        if theta is None:
            self._theta = np.zeros(self.n)
        else:
            if not isinstance(theta, tuple):
                raise TypeError('Theta should be given as a tuple')
            self._theta = np.array(theta)

        self._path = self.theta

        self.longest = np.argmax(lengths)
        self.annulus = self.lengths[self.longest] > np.sum(np.delete(self.lengths, self.longest))
        if self.annulus:
            # Can't reach points inside this radius
            self.inner_reach = self.lengths[self.longest] - np.sum(np.delete(self.lengths, self.longest))
        else:
            self.inner_reach = 0

    def generate_initial_guess(self, show=False, first_configuration=None):
        # How should the arm be positioned before generating a valid
        # configuration which moves the arm to the first destination
        if first_configuration is None:
            last_theta = self.theta
        else:
            assert first_configuration.shape == (self.n,)
            last_theta = first_configuration

        # Calculate initial guess for joint angles which hit all the
        # destinations but not necessarily with little effort
        self.initial_guess = np.empty((self.n, self.s,))
        assert self.initial_guess.shape == (self.n, self.s,)
        for index, destination in enumerate(self.destinations.T):
            last_theta = self.calculate_joint_angles(destination, last_theta)
            self.initial_guess[:, index] = last_theta

        fig = path_figure(self.initial_guess, self, show=False)
        fig.suptitle('Initial guess calculated by BFGS')
        if show is True:
            plt.show()

        return self.initial_guess.reshape((self.n * self.s,), order='F')

    def calculate_optimal_path(self, show=False):
        penalized_objective = generate_quadratically_penealized_objective(self)
        raise NotImplementedError

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    def calculate_joint_angles(self, p, theta=None):
        '''
        Takes in a destination p and an initial guess theta and returns a
        theta which maps closely to the point p
        '''
        # If theta is not provided, use the zero as initial guess
        if theta is None:
            theta = np.zeros(self.n)
        assert theta.ndim == 1

        # Check if the goal is out of reach
        p_length = np.linalg.norm(p)

        if p_length >= self.reach:
            print("Point outside interior of configuration space. Minimum found analytically.")
            return self.move_closest_to_out_of_reach(p)
        elif self.annulus and p_length <= self.inner_reach:
            print("Point outside interior of configuration space. Minimum found analytically.")
            return self.move_closest_to_out_of_reach(p, internal=True)
        else:
            f = self.generate_f(p)
            gradient = self.generate_gradient(p)
            return BFGS(theta, f, gradient, self.precision, initial_guess=True)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if value.shape == self._theta.shape:
            self._theta = value
        else:
            raise ValueError('New theta value does not match existing dimensions of theta. '
                             '{} != {}'.format(value.shape, self._theta.shape))

    def move_to(self, new_theta):
        self.theta = new_theta
        self.path = np.column_stack((self.path, self.theta))

    def move_closest_to_out_of_reach(self, p, internal=False):
        angle = np.math.atan2(p[1], p[0])
        theta = np.zeros(self.n)
        theta[0] = angle
        if internal is True:
            theta[0] += np.pi
            theta[self.longest] += np.pi
            if self.longest < self.n - 1:
                theta[self.longest + 1] += np.pi
        return theta

    def joint_positions(self, theta):
        '''
        Given a theta vector, this returns where the joints will be located.
        '''
        # The effective angles of each joint relative to x-axis
        joint_angles = np.cumsum(theta)

        # Components of each joint vector
        normalized_components = np.array([np.cos(joint_angles), np.sin(joint_angles)]).reshape((2, self.n))
        components = normalized_components * self.lengths.transpose()
        return np.cumsum(components, axis=1).reshape((2, self.n))

    def position(self, theta):
        if not theta.shape == (self.n,):
            raise ValueError('RobotArm got wrong dimensions of theta: ' + str(theta.shape))

        # The effective angles of each joint relative to x-axis
        joint_angles = np.cumsum(theta)

        # Components of each joint vector
        normalized_components = np.array([np.cos(joint_angles), np.sin(joint_angles)]).reshape((2, self.n))
        components = normalized_components * self.lengths.transpose()

        return np.sum(components, axis=1).reshape((2,))

    def generate_f(self, p):
        def f(theta):
            return np.linalg.norm(self.position(theta) - p) ** 2 / 2

        return f

    def generate_gradient(self, p):
        def gradient(theta):
            joint_angles = np.cumsum(theta)
            x_components = self.lengths * np.cos(joint_angles)
            y_components = self.lengths * np.sin(joint_angles)

            def a(i): return np.sum(x_components[i:], dtype=np.float)

            def b(i): return np.sum(y_components[i:], dtype=np.float)

            def f_partial_derivative(k):
                return np.float(-b(k) * (a(0) - p[0]) + a(k) * (b(0) - p[1]))

            return np.array([f_partial_derivative(k) for k in np.arange(0, self.n)], dtype=np.float)

        return gradient

    def _hessian(self):
        joint_angles = np.cumsum(self.theta)
        x_components = self.lengths * np.cos(joint_angles)
        y_components = self.lengths * np.sin(joint_angles)

        def a(i): return np.sum(x_components[i:])

        def b(i): return np.sum(y_components[i:])

        def f_second_derivative(k, l):
            beta = np.maximum(k, l)
            x_component = -a(beta) * (a(0) - self.p[0]) + b(k) * b(l)
            y_component = -b(beta) * (b(0) - self.p[1]) + a(k) * a(l)
            return x_component + y_component

        indices = np.arange(0, self.n)
        K, L = np.meshgrid(indices, indices)
        return np.vectorize(f_second_derivative)(K, L)

    # Optimization methods
    def f(self, theta, save=False):
        # TODO: Change API
        if save:
            self.path = np.column_stack((self.path, theta))
        else:
            self.theta = theta
            return self.remaining_distance()

    def hessian(self, theta):
        self.theta = theta
        return self._hessian()

    # Plotting stuff
    def plot_movement(self):
        joint_positions = self.position(joints=True)
        path = plotting.path_figure(joint_positions, destinations)
        path.show()

    def save_animation(self, fps=5, time=5, *, filename="robot_animation.gif"):
        original_position = self.theta
        number_of_frames = self.path.shape[1]

        frames_to_animate = fps * time
        if number_of_frames < frames_to_animate:
            step = 1
            fps = max(1, time // number_of_frames)
        else:
            step = number_of_frames // frames_to_animate

        fig = plt.figure()
        robot_animation = ImageMagickWriter(fps=fps)

        with robot_animation.saving(fig, filename, dpi=150):
            for column in np.arange(start=0, stop=number_of_frames, step=step):
                self.theta = np.array(self.path[:, column])
                self.plot(show=False)
                robot_animation.grab_frame()
                fig.clear()
                self._set_plot_options()

        self.theta = original_position  # ??
        plt.close()

    def plot_convergence(self):
        f_values = np.apply_along_axis(self.f, axis=0, arr=self.path)
        print(f_values)

        plt.figure()
        plt.plot(1 / 2 * np.log(2 * f_values))
        plt.xlabel('Iteration number')
        plt.ylabel('Logarithm of remaining distance')
        plt.title('Convergence plot')

        plt.show()


if __name__ == '__main__':
    lengths = (3, 2, 2,)
    destinations = (
        (5, 4, 6, 4, 5),
        (0, 2, 0.5, -2, -1),
    )
    theta = (np.pi, np.pi / 2, 0,)
    robot_arm = RobotArm(lengths, destinations, theta)
    robot_arm.generate_initial_guess(show=True)
