import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt


def generate_config_space_points(l, c, n_configs=2e4):
    n = np.size(l)
    n_angles = int(np.power(n_configs, 1 / n))

    angles1 = np.linspace(-c, c, num=n_angles)
    log_space = c - np.logspace(-3, np.log10(c), n_angles//2)
    angles2 = np.hstack((-log_space, log_space[:-1][::-1]))

    tuple_angles1 = tuple([np.hstack((angles1, angles2)) for _ in range(0, n)])
    tuple_angles2 = tuple([angles2 for _ in range(0, n)])
    thetas1 = np.array([np.meshgrid(*tuple_angles1)]).T.reshape(-1, n)
    thetas2 = np.array([np.meshgrid(*tuple_angles2)]).T.reshape(-1, n)
    thetas = np.vstack((thetas1))
    n_thetas = np.size(thetas, 0)

    points = np.zeros((n_thetas, 2))
    for i in range(0, n_thetas):
        points[i] = position(l, thetas[i])

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
    axis.scatter(points[:, 0], points[:, 1], color='grey', alpha=0.3)


def is_close_to_config_space(target_point, neighbor_tree, lengths):
    r = np.sum(lengths) / 100
    neighbors = neighbor_tree.query_ball_point(target_point, r)
    n_neighbors = np.size(neighbors)

    return n_neighbors > 10 * neighbor_tree.n / 1e5


if __name__ == '__main__':
    lengths = np.array([1, 3, 2])
    c = np.pi / 5

    target_point1 = np.array([1.2, 5.24])
    target_point2 = np.array([4.013, 3.695])
    points, neighbor_tree = generate_config_space_points(lengths, c)

    print(is_close_to_config_space(target_point1, neighbor_tree, lengths))
    print(is_close_to_config_space(target_point2, neighbor_tree, lengths))

    plt.figure()
    plt.style.use('ggplot')
    axis = plt.gca()

    plot_config_space(points, axis)
    axis.scatter(target_point1[0], target_point1[1], color='red')
    axis.scatter(target_point2[0], target_point2[1], color='red')
    plt.show()
