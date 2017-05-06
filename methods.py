import numpy as np

class MaximumIterationError(Exception):
    """Raise when exceeding predefined maximum number of iterations."""


def gradient_descent(x0, f, grad_f, epsilon, alpha0=1, c1=1e-4, rho=0.5):
    x = x0
    grad = grad_f(x0)
    norm_of_grad = np.linalg.norm(grad)

    it = 0
    max_it = 1e5
    while f(x) > epsilon:
        p = - grad / norm_of_grad
        alpha = armijo_backtracking_line_search(x, p, f, grad, alpha0, c1, rho)
        x += alpha * p + np.random.uniform(low=-1e-1, high=1e-1, size=len(x0))

        # if it % 100 == 0:
        #     f(x, save=True)

        grad = grad_f(x)
        norm_of_grad = np.linalg.norm(grad)

        it += 1
        if it == max_it:
            raise MaximumIterationError('Maximum iterations exceeded in function gradient_descent()')

    print('Gradient descent converged in {} iterations.'.format(it))
    return x


def BFGS(x0, f, grad_f, epsilon, alpha_steps=20, c1=1e-4, c2=0.9):
    x_k = x0
    grad_k = grad_f(x0)
    norm_of_grad_k = np.linalg.norm(grad_k)

    n = len(x0)
    I = np.identity(n)
    H_k = I  # initial inverse hessian approximation

    def phi_derivative(x, alpha, p):
        return np.inner(grad_f(x + alpha * p), p)

    first = True
    stationary_point_visits = 0
    max_it = int(1e5)
    for i in range(0, max_it):
        # Stop test
        if norm_of_grad_k < epsilon:
            print('BFGS converged in {} iterations.'.format(i))
            return x_k

        # if norm_of_grad_k < epsilon:
        #     print("BFGS found a stationary point after iteration {}.".format(i))
        #     stationary_point_visits += 1

        #     # Try to move away from stationary points that are not (global) minimizers
        #     number = 10 * n ** 2
        #     length = 2 ** stationary_point_visits * 4 / 180 * np.pi * np.sqrt(n)

        #     noise = np.random.uniform(low=-1.0, high=1.0, size=(n, number))
        #     noise = length * noise / np.linalg.norm(noise, axis=0)
        #     candidates = np.apply_along_axis(f, axis=0, arr=x_k.reshape(n, 1) + noise)
        #     best = np.argmin(candidates, axis=0)

        #     p_k = noise[:, best]
        #     alpha = min(np.linspace(0, 10, 1000), key=lambda a: f(x_k + a * p_k))

        #     # Reset BFGS
        #     x_k += alpha * p_k
        #     grad_k = grad_f(x_k)

        #     H_k = I  # initial inverse hessian approximation
        #     first = True

        else:
            # Calculate direction and stepsize
            p_k = - np.dot(H_k, grad_k)
            assert np.inner(p_k, grad_k) < 0
            alpha = wolfe_line_search(x_k, p_k, f, phi_derivative, alpha_steps, c1, c2)

            # Update values
            x_k_1 = x_k + alpha * p_k
            # f(x_k_1, save=True)
            grad_k_1 = grad_f(x_k_1)

            s_k = x_k_1 - x_k
            y_k = grad_k_1 - grad_k

            if first:
                H_k = np.inner(y_k, s_k) / np.inner(y_k, y_k) * I
                first = False

            assert not np.isclose(np.inner(y_k, s_k), 0)
            rho_k = 1 / np.inner(y_k, s_k)
            left_mat = I - rho_k * np.outer(s_k, y_k)
            right_mat = I - rho_k * np.outer(y_k, s_k)
            H_k = left_mat @ H_k @ right_mat + rho_k * np.outer(s_k, s_k)

            # Prepare for next step (k + 1)
            x_k = x_k_1
            grad_k = grad_k_1

        norm_of_grad_k = np.linalg.norm(grad_k)

    raise MaximumIterationError('Maximum iterations exceeded in function BFGS()')


def armijo_backtracking_line_search(x, p, f, grad, alpha, c1, rho):
    f_k = f(x)
    f_k_1 = f(x + alpha * p)

    it = 0
    max_it = 1e5
    while f_k_1 > f_k + c1 * alpha * np.inner(grad, p):
        alpha *= rho
        f_k_1 = f(x + alpha * p)

        it += 1
        if it == max_it:
                raise MaximumIterationError('Maximum iterations exceeded in function armijo_backtracking_line_search()')

    return alpha


def wolfe_line_search(x, p, f, phi_derivative, alpha_steps, c1, c2):
    alpha0 = 0
    alpha_i = alpha0
    alpha_i_1 = 1

    phi_0 = f(x)
    phi_alpha_i = phi_0
    phi_derivative_0 = phi_derivative(x, 0, p)

    for i in range(0, alpha_steps):
        phi_alpha_i_1 = f(x + p * alpha_i_1)

        if phi_alpha_i_1 > phi_0 + c1 * alpha_i_1 * phi_derivative_0 or \
                (phi_alpha_i_1 >= phi_alpha_i and i > 0):  # Wolfe 1 violated or alpha_i+1 > alpha_i
            return zoom(x, p, f, phi_derivative, phi_0, phi_derivative_0, alpha_i, alpha_i_1, c1, c2)

        phi_derivative_i_1 = phi_derivative(x, alpha_i_1, p)
        if np.abs(phi_derivative_i_1) <= -c2 * phi_derivative_0:  # Wolfe 2 satisfied, and Wolfe 1 (not violated)
            return alpha_i_1

        if phi_derivative_i_1 >= 0:  # Wolfe 2 violated
            return zoom(x, p, f, phi_derivative, phi_0, phi_derivative_0, alpha_i_1, alpha_i, c1, c2)

        alpha_i = alpha_i_1
        alpha_i_1 = 2 * alpha_i
        phi_alpha_i = phi_alpha_i_1

    raise MaximumIterationError('Maximum iterations exceeded in function wolfe_line_search()')


def zoom(x, p, f, phi_derivative, phi_0, phi_derivative_0, alpha_lo, alpha_hi, c1, c2):
    max_it = int(1e4)
    for i in range(0, max_it):
        alpha_j = (alpha_lo + alpha_hi) / 2
        phi_alpha_j = f(x + alpha_j * p)
        phi_alpha_lo = f(x + alpha_lo * p)

        if phi_alpha_j > phi_0 + c1 * alpha_j * phi_derivative_0 or phi_alpha_j >= phi_alpha_lo:
            alpha_hi = alpha_j

        else:
            phi_derivative_j = phi_derivative(x, alpha_j, p)

            if np.abs(phi_derivative_j) <= -c2 * phi_derivative_0:
                return alpha_j

            if phi_derivative_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo

            alpha_lo = alpha_j

    raise MaximumIterationError('Maximum iterations exceeded in function zoom()')
