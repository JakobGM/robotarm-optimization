import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt


def generate_config_space_points(lengths, c, n_configs=2e4):
    n = np.size(lengths)
    n_angles = int(np.power(n_configs, 1 / n))

    angles1 = np.linspace(-c, c, num=n_angles)
    log_space = c - np.logspace(-3, np.log10(c), n_angles//2)
    angles2 = np.hstack((-log_space, log_space[:-1][::-1]))

    tuple_angles = tuple([np.hstack((angles1, angles2)) for _ in range(0, n)])
    thetas = np.array([np.meshgrid(*tuple_angles)]).T.reshape(-1, n)
    n_thetas = np.size(thetas, 0)

    points = np.zeros((n_thetas, 2))
    for i in range(0, n_thetas):
        points[i] = position(lengths, thetas[i])

    neighbor_tree = scipy.spatial.cKDTree(points, leafsize=100)

    return points, neighbor_tree


def position(lengths, theta):
    n = np.size(lengths)
    cumulative_angles = np.cumsum(theta)

    # Components of each joint vector
    normalized_components = np.array([np.cos(cumulative_angles), np.sin(cumulative_angles)]).reshape((2, n))
    components = normalized_components * lengths.transpose()

    return np.sum(components, axis=1).reshape((2,))


def plot_config_space(points, axis):
    axis.scatter(points[:, 0], points[:, 1], color='grey', alpha=0.01)


def is_close_to_config_space(target_point, neighbor_tree, lengths):
    r = np.sum(lengths) / 50
    neighbors = neighbor_tree.query_ball_point(target_point, r)
    n_neighbors = np.size(neighbors)

    return n_neighbors > 5 * neighbor_tree.n / 1e5


if __name__ == '__main__':
    # lengths = np.array([1, 3, 2])
    # c = np.pi / 5
    # target_points = np.array([1.2, 5.24, 4.013, 3.695]).reshape(2, 2)

    lengths = np.array([3, 2, 2])
    c = np.pi / 2
    target_points = np.array([-1, 5, -3, 3, -3, -4, 0, 5, 3, 2, 0, 7.1]).reshape(6, 2) # 0.0785, 3.565

    points, neighbor_tree = generate_config_space_points(lengths, c)

    plt.figure()
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    axis = plt.gca()

    plot_config_space(points, axis)

    for i in range(0, target_points.shape[0]):
        print(is_close_to_config_space(target_points[i], neighbor_tree, lengths))
        axis.scatter(target_points[i, 0], target_points[i, 1], color='red')

    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # plt.savefig('figures/config_space_angular_constraints.png', bbox_inches='tight', dpi=500)
    plt.show()
