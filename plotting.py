from matplotlib import pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
from config_space_angular_constraints import plot_config_space


def path_figure(theta_matrix, robot_arm, show=True):
    """
    Arguments:
    theta_matrix - A set of theta column vectors
    robot_arm - An object of the RobotArm class

    Returns:
    None, but plots the configuration of each theta vector as subplots
    """
    # Check input arguments
    num_of_destinations = robot_arm.destinations.shape[1]
    if not theta_matrix.shape == (robot_arm.n, num_of_destinations):
        raise ValueError('''
                        The number of joint positions does not match the
                         number of destination points
                         ''')

    # Set up plot style options
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=2, ncols=3)
    np.ravel(axes)[-1].axis('off')
    axes = np.ravel(axes)[:-1]
    for ax in np.ravel(axes):
        set_axis_options(ax, robot_arm)

    # Plotting content of each subplot
    for index, theta in enumerate(theta_matrix.T):
        plot_position(np.ravel(axes)[index], theta, robot_arm)

    if show is True:
        #plt.savefig('figures/inequality.png', bbox_inches='tight', dpi=500)
        plt.show()

    return fig


def set_axis_options(ax, robot_arm):
    ax.set_autoscale_on(False)
    ax.set_aspect('equal')
    if robot_arm.angular_constraint is not None:
        plot_config_space(robot_arm.config_space_points, ax)

    ax.axhline(y=0, color='grey')
    ax.axvline(x=0, color='grey')

    # Padding
    a = 1.1
    max_x = abs(max(robot_arm.destinations, key=lambda p: abs(p[0]))[0])
    max_y = abs(max(robot_arm.destinations, key=lambda p: abs(p[1]))[1])
    m = max(max_x, max_y, robot_arm.reach)

    ax.set_xlim(0, a * m)
    ax.set_ylim(-a * m, a * m)


def plot_position(axis, theta, robot_arm):
    joint_positions = robot_arm.joint_positions(theta)
    x = np.hstack((0, joint_positions[0, :]))
    y = np.hstack((0, joint_positions[1, :]))
    axis.plot(x, y, '-o')

    # Plot all the points that shall be reached
    for index, p in enumerate(robot_arm.destinations.T):
        point, = axis.plot(p[0], p[1], 'x')
        axis.text(p[0], p[1], str(index + 1), fontsize=14, color=point.get_color())

    # Plot configuration space of robot
    if robot_arm.angular_constraint is None:
        configuration_space = Wedge(
            (0, 0),
            r=robot_arm.reach,
            theta1=0,
            theta2=360,
            width=robot_arm.reach - robot_arm.inner_reach,
            facecolor='grey',
            alpha=0.3,
            edgecolor='black',
            linewidth=0.6
        )

        axis.add_patch(configuration_space)
