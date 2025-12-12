import numpy as np
from nulapack import gauss_seidel


def test_single_precision():
    A = np.array(
        [
            [10.0, -1.0, 2.0, 0.0],
            [-1.0, 11.0, -1.0, 3.0],
            [2.0, -1.0, 10.0, -1.0],
            [0.0, 3.0, -1.0, 8.0],
        ],
        dtype=np.float32,
    )
    B = np.array([6.0, 25.0, -11.0, 15.0], dtype=np.float32)

    X, status = gauss_seidel(A, B, max_iter=1000, tol=1e-4, omega=1.0)

    assert status == 0
    assert np.allclose(X, [1.0, 2.0, -1.0, 1.0], atol=1e-5)


def test_double_precision():
    A = np.array(
        [
            [10.0, -1.0, 2.0, 0.0],
            [-1.0, 11.0, -1.0, 3.0],
            [2.0, -1.0, 10.0, -1.0],
            [0.0, 3.0, -1.0, 8.0],
        ],
        dtype=np.float64,
    )
    B = np.array([6.0, 25.0, -11.0, 15.0], dtype=np.float64)

    X, status = gauss_seidel(A, B, max_iter=1000, tol=1e-10, omega=1.0)

    assert status == 0
    assert np.allclose(X, [1.0, 2.0, -1.0, 1.0], atol=1e-10)


def test_complex_float():
    A = np.array(
        [
            [10.0 + 1.0j, -1.0, 2.0, 0.0],
            [-1.0, 11.0 + 1.0j, -1.0, 3.0],
            [2.0, -1.0, 10.0 + 1.0j, -1.0],
            [0.0, 3.0, -1.0, 8.0 + 1.0j],
        ],
        dtype=np.complex64,
    )
    B = np.array(
        [6.0 + 1.0j, 25.0 + 2.0j, -11.0 + 1.0j, 15.0 - 1.0j], dtype=np.complex64
    )

    X, status = gauss_seidel(A, B, max_iter=1000, tol=1e-4, omega=1.0)

    assert status == 0

    expected = np.array(
        [
            0.995412 - 0.028752j,
            2.018524 + 0.081610j,
            -0.982175 + 0.186858j,
            0.963693 - 0.252708j,
        ],
        dtype=np.complex64,
    )

    assert np.allclose(X, expected, atol=1e-5)


def test_complex_double():
    A = np.array(
        [
            [10.0 + 1.0j, -1.0, 2.0, 0.0],
            [-1.0, 11.0 + 1.0j, -1.0, 3.0],
            [2.0, -1.0, 10.0 + 1.0j, -1.0],
            [0.0, 3.0, -1.0, 8.0 + 1.0j],
        ],
        dtype=np.complex128,
    )
    B = np.array(
        [6.0 + 1.0j, 25.0 + 2.0j, -11.0 + 1.0j, 15.0 - 1.0j], dtype=np.complex128
    )

    X, status = gauss_seidel(A, B, max_iter=1000, tol=1e-12, omega=1.0)

    assert status == 0

    expected = np.array(
        [
            0.995412230296 - 0.028751867347j,
            2.018524356148 + 0.081609612259j,
            -0.982174907078 + 0.186858027715j,
            0.963693005950 - 0.252707976877j,
        ],
        dtype=np.complex128,
    )

    assert np.allclose(X, expected, atol=1e-11)
