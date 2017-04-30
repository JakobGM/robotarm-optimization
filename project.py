import numpy as np
from methods_project_old import gradient_descent, newtons_method, BFGS, MaximumIterationError
from robot_arm import RobotArm

if __name__ == '__main__':
    lengths, theta0, p = np.load('initial_values_bug.npy')

    WALL_E = RobotArm(lengths, destination=p, theta=theta0, precision=1e-3)

    try:
        WALL_E.move_to_destination()
    except MaximumIterationError:
        WALL_E.save_animation()
        raise
    except AssertionError:
        WALL_E.save_animation()
        raise

    # WALL_E.save_animation()
    WALL_E.plot()
    print(WALL_E.position())
