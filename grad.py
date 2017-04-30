import  numpy as np

def grad_f(theta, lengths, p):
    n = len(lengths)

    joint_angles = np.cumsum(theta)
    x_components = lengths * np.cos(joint_angles)
    y_components = lengths * np.sin(joint_angles)

    def a(i): return np.sum(x_components[i:])

    def b(i): return np.sum(y_components[i:])

    def f_partial_derivative(k):
        return -b(k) * (a(0) - p[0]) + a(k) * (b(0) - p[1])

    return np.array([f_partial_derivative(k) for k in np.arange(0, n)]).reshape(n, 1)

theta = np.array([1, 0, np.pi])
lengths = np.array([3, 2, 2])
p = [np.cos(1), np.sin(1)]

print(grad_f(theta, lengths, p))