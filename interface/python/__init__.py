import _nulapack
import numpy as np

__version__ = "0.1.0"
__all__ = ["gauss_seidel"]


def gauss_seidel(A, B, max_iter=1000, tol=1e-8, omega=1.0):
    """
    Solve the linear system Ax = B using the Gauss-Seidel method.

    Parameters
    ----------
    A : ndarray
        Coefficient matrix (n x n)
    B : ndarray
        Right-hand side vector (n,)
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Convergence tolerance
    omega : float, optional
        Relaxation factor (default is 1.0, standard Gauss-Seidel)

    Returns
    -------
    X : ndarray
        Solution vector
    status : int
        0 if converged, non-zero otherwise
    """
    A = np.ascontiguousarray(A)
    B = np.asfortranarray(B)
    n = A.shape[0]

    X = np.zeros_like(B)

    A_flat = A.ravel()

    if np.issubdtype(A.dtype, np.floating):
        if A.dtype == np.float32:
            status = _nulapack.sgssv(A_flat, B, X, max_iter, tol, omega, 0, n)
        else:  # float64
            status = _nulapack.dgssv(A_flat, B, X, max_iter, tol, omega, 0, n)
    elif np.issubdtype(A.dtype, np.complexfloating):
        if A.dtype == np.complex64:
            status = _nulapack.cgssv(A_flat, B, X, max_iter, tol, omega, 0, n)
        else:  # complex128
            status = _nulapack.zgssv(A_flat, B, X, max_iter, tol, omega, 0, n)
    else:
        raise TypeError("Unsupported array dtype: {}".format(A.dtype))

    return X, int(status) if status is not None else 0
