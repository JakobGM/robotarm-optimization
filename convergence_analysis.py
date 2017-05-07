import numpy as np
from matplotlib import pyplot as plt

from constraints import generate_constraints_function

def plot_convergence(xs, objective, robot_arm, show=False):
    """
    A function for plotting the convergence of an optimization algorithm.
    Input:
    xs        - The inputs generated iteratively by the optimization method,
                given as column vectors in an n times i array, where n is the
                dimension of the domain space, and i is the numbers of
                iterations performed by the optimization method.
    objective - The function to be minimized by the optimization method
    """
    # Make sure the xs are never mutated (because I'm a bad programmer)
    xs.setflags(write=False)

    # Dimension of domain space and number of method iterations
    n, i = xs.shape

    # The final solution of the method is used as a numerical refernce
    # solution
    minimizer = xs[:, -1]
    minimum = objective(minimizer)

    # Calculate remaining distance to final minimizer
    remaining_distance = xs - minimizer.reshape((n, 1,))
    remaining_distance = np.linalg.norm(
        remaining_distance,
        ord=2,
        axis=0
    )
    assert remaining_distance.shape == (i,)

    # Calculate the decrement in the objective values
    objective_values = np.apply_along_axis(
        objective,
        axis=0,
        arr=xs
    )
    remaining_decline = objective_values - minimum
    assert remaining_decline.shape == (i,)

    # Calculate the change in constraint values over time
    constraints_func = generate_constraints_function(robot_arm)
    constraints_values = np.apply_along_axis(
        constraints_func,
        axis=0,
        arr=xs
    )
    print('Constraint_values; ', constraints_values)
    constraints_values = np.sum(np.abs(constraints_values), axis=0)

    # Create three subplots, one showing convergence of the minimizer,
    # the other showing the convergence to the mimimum (input vs. output),
    # and the third the values of the constraints over time
    plt.style.use('ggplot')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot minimizer convergence
    ax = plt.subplot('131')
    ax.plot(remaining_distance[:-1])
    ax.set_yscale('log')
    ax.set_title('Convergence towards \n optimal configuration')
    ax.set_ylabel(r'$||\Theta - \Theta^*||$')
    ax.set_xlabel(r'Iteration number')

    # Plot minimum convergence
    ax = plt.subplot('132')
    ax.plot(objective_values)
    ax.set_title('Objective values')
    ax.set_ylabel(r'$E(\Theta)$')
    ax.set_xlabel(r'Iteration number')

    # Plot values of constraints
    ax = plt.subplot('133')
    ax.plot(constraints_values)
    ax.set_title('Equality constraint values')
    ax.set_ylabel(r'$\sum_{i=1}^{s}(|c_i,x(\Theta)| + |c_i,y(\Theta)|)$')
    ax.set_xlabel(r'Iteration number')

    fig = plt.gcf()
    fig.set_size_inches(18.5, 5)
    plt.savefig('figures/equality_constraint.pdf', bbox_inches='tight')

    if show is True:
        plt.show()
