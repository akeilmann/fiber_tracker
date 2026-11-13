import numpy as np


def axial_mean(vectors, eps=1e-8):
    """Mean direction of axial data, where v and -v describe the same line.

    Returns the principal eigenvector of the orientation tensor
    T = mean(v v^T) over the unit vectors v (Advani & Tucker 1987), and its
    eigenvalue as a measure of agreement: 1 = all vectors parallel,
    1/3 = no preferred direction (3D). The sign of the result is arbitrary.
    Returns (None, 0.0) if there are no non-zero vectors.
    """
    v = np.asarray(vectors, np.float64).reshape(-1, 3)
    n = np.linalg.norm(v, axis=1)
    v = v[n > eps] / n[n > eps, None]
    if len(v) == 0:
        return None, 0.0
    w, e = np.linalg.eigh(v.T @ v / len(v))      # eigenvalues in ascending order
    return e[:, -1], w[-1]
