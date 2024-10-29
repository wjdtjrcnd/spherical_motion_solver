import numpy as np
import polynomial_solver as ps
import action_matrix_solver as ms
def main():
    # Make random 3D points
    X = np.random.rand(3, 3) * 2 - 1

    # Make random rotation and translation
    R = np.random.rand(3, 3) * 2 - 1
    U, S, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        R = -R
    # Outward-facing
    t = R[:, 2] - np.array([0, 0, 1])

    # Get points in second camera
    PX = R @ X + t[:, np.newaxis]

    # Get true E
    t_x = np.array([[0, -t[2], t[1]],
                    [t[2], 0, -t[0]],
                    [-t[1], t[0], 0]])
    E = t_x @ R
    E /= np.linalg.norm(E.flatten())

    # Get projections in cameras
    u = X / X[2, :]
    v = PX / PX[2, :]

    # Get solutions
    Esolns_AM = ms.solve_spherical_action_matrix(u, v)
    Esolns_poly = ps.solve_spherical_polynomial(u, v)

    # Get error in Frobenius norm
    error_AM = np.zeros(4)
    error_poly = np.zeros(4)

    for i in range(4):
        err1 = E - Esolns_AM[:, :, i]
        err2 = E + Esolns_AM[:, :, i]
        error_AM[i] = min(np.linalg.norm(err1.flatten()), np.linalg.norm(err2.flatten()))

        err1 = E - Esolns_poly[:, :, i]
        err2 = E + Esolns_poly[:, :, i]
        error_poly[i] = min(np.linalg.norm(err1.flatten()), np.linalg.norm(err2.flatten()))

    # Display errors
    print('Action matrix error: ', end='')
    print(' '.join(f'{err:.2f}' for err in error_AM))
    print('Polynomial error: ', end='')
    print(' '.join(f'{err:.2f}' for err in error_poly))

if __name__ == "__main__":
    main()
