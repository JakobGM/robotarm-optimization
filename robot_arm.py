import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ImageMagickWriter
from matplotlib.patches import Wedge
from methods_project_old import BFGS


class RobotArm:
    def __init__(self, lengths, destination, theta=None, precision=1e-2):
        assert lengths.ndim == 1
        assert len(destination) == 2
        self.lengths = lengths
        self.reach = np.sum(lengths)
        self.n = len(lengths)
        self.p = destination
        self.precision = precision

        if theta is None:
            self._theta = np.zeros(self.n)
        else:
            assert theta.ndim == 1
            self._theta = theta

        self._path = self.theta

        self._set_plot_options()

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

    def plot(self, show=True):
        joint_positions = self.position(joints=True)
        x = np.hstack((0, joint_positions[0, :]))
        y = np.hstack((0, joint_positions[1, :]))

        self._set_plot_options()

        plt.plot(x, y, '-o')
        plt.plot(self.p[0], self.p[1], 'x')

        # Plot configuration space of robot
        configuration_space = Wedge((0, 0), r=self.reach, theta1=0, theta2=360, width=self.reach - self.inner_reach,
                                    facecolor='grey', alpha=0.3, edgecolor='black', linewidth=0.6)

        ax = plt.gca()
        ax.add_patch(configuration_space)

        if show is True:
            plt.show()

    def _set_plot_options(self):
        plt.style.use('ggplot')

        ax = plt.gca()
        ax.set_autoscale_on(False)

        ax.axhline(y=0, color='grey')
        ax.axvline(x=0, color='grey')

        # Padding
        a = 1.1
        m = max(abs(self.p[0]), abs(self.p[1]), self.reach)

        ax.set_xlim(-a * m, a * m)
        ax.set_ylim(-a * m, a * m)

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
    # theta = np.array([np.pi / 4, np.pi / 4, np.pi / 3]).reshape((3, 1))
    # lengths = np.ones(3).reshape((3, 1))
    # p = np.array([1, 2]).reshape(2,1)
    # EVE = RobotArm(lengths, p, theta)
    #
    # theta_sequence = np.zeros((3, 100))
    # theta_sequence[:, 0] = theta[:, 0]
    # for n in range(99):
    #     theta_sequence[:, n + 1] = theta_sequence[:, n]
    #     theta_sequence[0, n] = 2 * np.pi / 100 * n
    #     theta_sequence[1, n] = np.pi / 25 * n
    #     EVE.move_to(theta_sequence[:, n].reshape(3, 1))
    #
    # EVE.save_animation()
    WALLE = RobotArm(np.ones(3), destination=np.array([2, 2]))