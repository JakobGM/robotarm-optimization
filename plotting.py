from matplotlib import pyplot as plt
import numpy as np


def path_figure(joint_positions_matrix, robot_arm):
    """
    Arguments:
    joint_positions_matrix - A
    TODO: Write docstring
    """
    # Check input arguments
    if not joint_positions_matrix.shape[1] == robot_arm.destinations.shape[1]:
        raise ValueError('''
                        The number of joint positions does not match the
                         number of destination points
                         ''')
    num_of_destinations = destinations.shape[1]

    # Set up plot style options
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=1, ncols=num_of_destinations)
    for ax in np.ravel(axes):
        set_axis_options(ax, robot_arm)

    # Plotting content of each subplot
    pass

def set_axis_options(ax, robot_arm):
    ax.set_autoscale_on(False)

    ax.axhline(y=0, color='grey')
    ax.axvline(x=0, color='grey')

    # Padding
    a = 1.1
    max_x = abs(max(robot_arm.destinations, key=lambda p: abs(p[0]))[0])
    max_y = abs(max(robot_arm.destinations, key=lambda p: abs(p[1]))[1])
    m = max(max_x, max_y, robot_arm.reach)

    ax.set_xlim(-a * m, a * m)
    ax.set_ylim(-a * m, a * m)

def position_fig(joint_positions, destination):
    pass



def plot(joint_positions, destinations, reach, inner_reach):
    joint_positions = position(joints=True)
    x = np.hstack((0, joint_positions[0, :]))
    y = np.hstack((0, joint_positions[1, :]))

    set_plot_options()

    # Plot all the points that shall be reached
    for p in destinations:
        plt.plot(x, y, '-o')
        plt.plot(p[0], p[1], 'x')

    # Plot configuration space of robot
    configuration_space = Wedge((0, 0), r=reach, theta1=0, theta2=360, width=reach - inner_reach,
                                facecolor='grey', alpha=0.3, edgecolor='black', linewidth=0.6)

    ax = plt.gca()
    ax.add_patch(configuration_space)

    if show is True:
        plt.show()
