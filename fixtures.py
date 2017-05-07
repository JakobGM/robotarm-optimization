"""
Fixtures for variables that need to be used across several tests
"""
import numpy as np

from robot_arm import RobotArm

lengths = (3, 2, 2,)
lengths2 = (3, 1, 1, 1, 1,)
destinations = (
    (5, 4, 6, 4, 5),
    (0, 2, 0.5, -2, -1),
)
theta = (np.pi, np.pi / 2, 0,)
robot_arm1 = RobotArm(lengths, destinations, theta)
robot_arm2 = RobotArm(lengths2, destinations)
