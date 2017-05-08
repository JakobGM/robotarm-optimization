# Robot arm movement algorithm written in Python with Numpy

Project task in Optimization 1 at NTNU. How to move a robot arm between consecutive points in two dimensions with minimum effort. This problem is solved with optimization techniques.

Results for the equality-constrained problem can be shown by simply running augmented_lagrangian.py.
This should produce a visual representation of the optimal configuration. Tweaking parameters in generate_simple_augmented_lagrangian_function() and
augmented_lagrangian_method should impact the performance of the method. Note that some extreme choices of parameters could cause the method to break down.
The fundamentals of the problem, such as segment lengths, can be changed by initializing the RobotArm object with other parameters.

Results for the inequality-constrained problem can be shown my simply running primal_log_barrier.py. This should produce a plot showing the solution
together with the associated configuration space. Tweaking parameters can also be done here, but the method is very sensitive to changes in both
the problem and to parameters related to the method.
