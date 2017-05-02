import plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ImageMagickWriter
from matplotlib.patches import Wedge
from methods import BFGS


def objective(thetas):
    assert isinstance(thetas, np.ndarray)
    rotated = np.roll(thetas.copy(), shift=-1, axis=1)
    deltas = rotated - thetas
    return 0.5 * np.sum(deltas ** 2)


def constraint(thetas, lengths, constraint_number):
    n = len(lengths)
    col_index = None
    if constraint_number % 2 == 1:
        col_index = (constraint_number + 1) // 2 - 1
    elif constraint_number % 2 == 0:
        col_index = constraint_number // 2 - 1
    theta_cum_sum = np.cumsum(thetas[:, col_index])
    constraint_value = 0
    for i in range(n):
        if constraint_number % 2 == 0:
            constraint_value += lengths(i) * np.sin(theta_cum_sum[i])
        elif constraint_number % 2 == 1:
            constraint_value += lengths(i) * np.cos(theta_cum_sum[i])
    return constraint_value


def objective_gradient(thetas):
    assert isinstance(thetas, np.ndarray)
    n = thetas.shape[0]
    s = thetas.shape[1]
    objective_gradient_matrix = np.zeros((n, s))
    for j in range(s):
        for i in range(n):
            if (j > 0) and j < (s - 1):
                objective_gradient_matrix[i, j] = 2 * thetas[i, j] - (thetas[i, j - 1] + thetas[i, j + 1])
            elif j == 0:
                objective_gradient_matrix[i, j] = 2 * thetas[i, j] - (thetas[i, s - 1] + thetas[i, 1])
            elif j == s - 1:
                objective_gradient_matrix[i, j] = 2 * thetas[i, j] - (thetas[i, s - 2] + thetas[i, 0])
    return objective_gradient_matrix


def constraint_gradient(thetas, lengths, constraint_number):
    assert isinstance(thetas, np.ndarray)
    n = thetas.shape[0]
    s = thetas.shape[1]
    col_index = None
    assert n == len(lengths)
    constraint_gradient_matrix = np.zeros((n, s))
    if constraint_number % 2 == 1:
        col_index = (constraint_number + 1) // 2 - 1
    elif constraint_number % 2 == 0:
        col_index = constraint_number // 2 - 1
    theta_cum_sum = np.cumsum(thetas[:, col_index])
    theta_cum_prod = np.cumprod(thetas[:, col_index])
    for i in range(n):
        elements_in_outer_sum = np.zeros(n)
        for j in range(n):
            elements_in_outer_sum[j] = lengths[j] * theta_cum_sum[j] / thetas[i, col_index]
            if constraint_number % 2 == 1:
                elements_in_outer_sum[j] *= -np.sin(theta_cum_prod[j])
            elif constraint_number % 2 == 0:
                elements_in_outer_sum[j] *= np.cos(theta_cum_prod[j])
        constraint_gradient_matrix[i, col_index] = -np.sum(elements_in_outer_sum[i:])
    return constraint_gradient_matrix


def constraint_grad_set(thetas, lengths):
    n = thetas.shape[0]
    s = thetas.shape[1]
    constraint_set = np.zeros((n * s, 2 * s))
    for i in range(1, 2 * s + 1):
        constraint_grad = constraint_gradient(thetas, lengths, i).flatten('F').reshape((n * s,))
        constraint_set[:, i - 1] = constraint_grad
    return constraint_set


class RobotArm:
    def __init__(self, lengths, destinations, theta=None, precision=1e-2):
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
        self.precision = precision

        if theta is None:
            self._theta = np.zeros(self.n)
        else:
            if not isinstance(theta, tuple):
                raise TypeError('Theta should be given as a tuple')
            self._theta = theta

        self._path = self.theta

        self.longest = np.argmax(lengths)
        self.annulus = self.lengths[self.longest] > np.sum(np.delete(self.lengths, self.longest))
        if self.annulus:
            # Can't reach points inside this radius
            self.inner_reach = self.lengths[self.longest] - np.sum(np.delete(self.lengths, self.longest))
        else:
            self.inner_reach = 0

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    def move_to_destination(self):
        # Check if the goal is out of reach
        p_length = np.linalg.norm(self.p)

        if p_length >= self.reach:
            print("Point outside interior of configuration space. Minimum found analytically.")
            self.move_closest_to_out_of_reach()
        elif self.annulus and p_length <= self.inner_reach:
            print("Point outside interior of configuration space. Minimum found analytically.")
            self.move_closest_to_out_of_reach(internal=True)
        else:
            raise NotImpelmentedError
            BFGS(self.theta, self.f, self.gradient, self.precision)

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

    def move_closest_to_out_of_reach(self, internal=False):
        angle = np.math.atan2(self.p[1], self.p[0])
        theta = np.zeros(self.n)
        theta[0] = angle
        if internal is True:
            theta[0] += np.pi
            theta[self.longest] += np.pi
            if self.longest < self.n - 1:
                theta[self.longest + 1] += np.pi
        self.move_to(theta)

    def joint_positions(self, theta):
        '''
        Given a theta vector, this returns where the joints will be located.
        '''
        # The effective angles of each joint relative to x-axis
        joint_angles = np.cumsum(self.theta)

        # Components of each joint vector
        normalized_components = np.array([np.cos(joint_angles), np.sin(joint_angles)]).reshape((2, self.n))
        components = normalized_components * self.lengths.transpose()
        return np.cumsum(components, axis=1).reshape((2, self.n))

    def position(self, joints=False):
        # The effective angles of each joint relative to x-axis
        joint_angles = np.cumsum(self.theta)

        # Components of each joint vector
        normalized_components = np.array([np.cos(joint_angles), np.sin(joint_angles)]).reshape((2, self.n))
        components = normalized_components * self.lengths.transpose()

        if joints is True:
            return np.cumsum(components, axis=1).reshape((2, self.n))
        else:
            return np.sum(components, axis=1).reshape((2,))

    def remaining_distance(self):
        return np.linalg.norm(self.position() - self.p) ** 2 / 2

    def _gradient(self):
        joint_angles = np.cumsum(self.theta)
        x_components = self.lengths * np.cos(joint_angles)
        y_components = self.lengths * np.sin(joint_angles)

        def a(i): return np.sum(x_components[i:], dtype=np.float)

        def b(i): return np.sum(y_components[i:], dtype=np.float)

        def f_partial_derivative(k):
            return np.float(-b(k) * (a(0) - self.p[0]) + a(k) * (b(0) - self.p[1]))

        return np.array([f_partial_derivative(k) for k in np.arange(0, self.n)], dtype=np.float)

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

    def gradient(self, theta):
        self.theta = theta
        return self._gradient()

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
