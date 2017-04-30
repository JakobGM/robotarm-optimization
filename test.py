import numpy as np
from methods_project_old import MaximumIterationError
from robot_arm import RobotArm

def test_point_outside_outer_circle(lengths, n, plot_initial=True, plot_minimizer=True, animate=False):
    print('---- Test with destination outside configuration space ----')
    analytical_tests('ooc', lengths, n, plot_initial, plot_minimizer, animate)


def test_point_innside_inner_circle(lengths, n, plot_initial=True, plot_minimizer=True, animate=False):
    # Precondition: Configuration space is an annulus
    longest = np.argmax(lengths)
    if np.sum(np.delete(lengths, longest)) > lengths[longest]:
        print("---- Configuration space not an annulus. Can't run test ----")
        return

    print("---- Test with destination innside inner circle of configuration space ----")
    analytical_tests('iic', lengths, n, plot_initial, plot_minimizer, animate)


def analytical_tests(test, lengths, n, plot_initial, plot_minimizer, animate):
    if test == 'iic':
        longest = np.argmax(lengths)
        inner_radius = lengths[longest] - np.sum(lengths)
        p_distance_from_origin = inner_radius / 2
    elif test == 'ooc':
        reach = np.sum(lengths)
        p_distance_from_origin = reach + 1
    else:
        print ("Test not implemented.")
        raise NotImplementedError

    angle = 2 * np.pi * np.random.random()
    p = p_distance_from_origin * np.array([np.cos(angle), np.sin(angle)])

    theta0 = 2 * np.pi * np.random.random(n)

    run_test(lengths, theta0, p, plot_initial, plot_minimizer, animate)


def test_bfgs_local_max(lengths, n, plot_initial=True, plot_minimizer=True, animate=False):
    print('---- Test with a boundary point that is a \n'
          'local (not global) maximum as starting point ----')
    bfgs_tests('lm', lengths, n, plot_initial, plot_minimizer, animate)


def test_bfgs_global_max(lengths, n, plot_initial=True, plot_minimizer=True, animate=False):
    print('---- Test with global maximum as starting point ----')
    bfgs_tests('gm', lengths, n, plot_initial, plot_minimizer, animate)


def test_bfgs_saddle_point(lengths, n, plot_initial=True, plot_minimizer=True, animate=False):
    print('---- Test with an interior point that is either a\n'
          ' saddle point or local maximum as starting point ----')
    bfgs_tests('sp', lengths, n, plot_initial, plot_minimizer, animate)


def bfgs_tests(test, lengths, n, plot_initial, plot_minimizer, animate):
    first_angle = 2 * np.pi * np.random.random()
    if test == 'sp': last_angle = np.pi
    theta0 = np.zeros(n)
    theta0[0] = first_angle

    longest = np.argmax(lengths)
    annulus = np.sum(np.delete(lengths, longest)) < lengths[longest]
    if annulus:
        if test == 'sp':
            if longest != len(lengths) - 1: theta0[-1] = last_angle
            else: theta0[-2] = last_angle

        inner_radius = lengths[longest] - np.sum(np.delete(lengths, longest))
        outer_radius = np.sum(lengths)
        p_distance_from_origin = inner_radius + (outer_radius - inner_radius) * np.random.random()
    else:
        p_distance_from_origin = np.sum(lengths) * np.random.random()

    if test == 'lm' or test == 'sp':
        p = p_distance_from_origin * np.array([np.cos(first_angle), np.sin(first_angle)])
    elif test == 'gm':
        p = p_distance_from_origin * np.array([-np.cos(first_angle), -np.sin(first_angle)])
    else:
        print ("Test not implemented.")
        raise NotImplementedError

    run_test(lengths, theta0, p, plot_initial, plot_minimizer, animate)


def test_random(m, plot_initial=False, plot_minimizer=False, animate=False):
    print('---- Test with m random configurations ----')

    n_cap = 100
    l_cap = 1000

    for i in range(0, m):
        n = int(n_cap * np.random.random()) + 1
        lengths = l_cap * np.random.random(n)
        theta0 = 2 * np.pi * np.random.random(n)

        p_distance_to_origin = 2 * np.sum(lengths) * np.random.uniform(low=-1, high=1)
        p_angle = 2 * np.pi * np.random.random()
        p = p_distance_to_origin * np.array([np.cos(p_angle), np.sin(p_angle)])

        run_test(lengths, theta0, p, plot_initial, plot_minimizer, animate)


def run_test(lengths, theta0, p, plot_initial, plot_minimizer, animate):
    WALL_E = RobotArm(lengths, p, theta0, precision=epsilon)
    if plot_initial: WALL_E.plot()

    try:
        WALL_E.move_to_destination()
    except MaximumIterationError:
        np.save('initial_values_bug', (lengths, theta0, p))
        WALL_E.save_animation()
        raise
    except AssertionError:
        np.save('initial_values_bug', (lengths, theta0, p))
        WALL_E.save_animation()
        raise

    if plot_minimizer: WALL_E.plot()
    if animate: WALL_E.save_animation()

if __name__ == '__main__':
    arms = []
    arms.append(np.array([3, 2, 2]))  # disk-shaped configuration space
    arms.append(np.array([1, 4, 1]))  # annulus-shaped configuration space

    epsilon = 1e-3

    k = len(arms)
    for lengths in arms:
        n = len(lengths)
        test_point_outside_outer_circle(lengths, n)
        test_point_innside_inner_circle(lengths, n)
        test_bfgs_local_max(lengths, n)
        test_bfgs_global_max(lengths, n)
        test_bfgs_saddle_point(lengths, n)

    test_random(100)

