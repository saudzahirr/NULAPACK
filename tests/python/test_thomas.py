import numpy as np
from nulapack import thomas


def _test_single_precision():
    a = np.array(
        [
            [10.0, -1.0, 0.0, 0.0],
            [-1.0, 11.0, -1.0, 0.0],
            [0.0, -1.0, 10.0, -1.0],
            [0.0, 0.0, -1.0, 8.0],
        ],
        dtype=np.float32,
    )
    b = np.array([6.0, 25.0, -11.0, 15.0], dtype=np.float32)

    x, info = thomas(a, b)
    assert info == 0

    residual = a @ x - b
    assert np.allclose(residual, 0, atol=1e-5)


def _test_double_precision():
    a = np.array(
        [
            [10.0, -1.0, 0.0, 0.0],
            [-1.0, 11.0, -1.0, 0.0],
            [0.0, -1.0, 10.0, -1.0],
            [0.0, 0.0, -1.0, 8.0],
        ],
        dtype=np.float64,
    )
    b = np.array([6.0, 25.0, -11.0, 15.0], dtype=np.float64)

    x, info = thomas(a, b)
    assert info == 0

    residual = a @ x - b
    assert np.allclose(residual, 0, atol=1e-12)


def _test_complex_float():
    a = np.array(
        [
            [10.0 + 1.0j, -1.0, 0.0, 0.0],
            [-1.0, 11.0 + 1.0j, -1.0, 0.0],
            [0.0, -1.0, 10.0 + 1.0j, -1.0],
            [0.0, 0.0, -1.0, 8.0 + 1.0j],
        ],
        dtype=np.complex64,
    )
    b = np.array([6.0 + 1.0j, 25.0 + 2.0j, -11.0 + 1.0j, 15.0 - 1.0j], dtype=np.complex64)

    x, info = thomas(a, b)
    assert info == 0

    residual = a @ x - b
    assert np.allclose(residual, 0, atol=1e-5)


def _test_complex_double():
    a = np.array(
        [
            [10.0 + 1.0j, -1.0, 0.0, 0.0],
            [-1.0, 11.0 + 1.0j, -1.0, 0.0],
            [0.0, -1.0, 10.0 + 1.0j, -1.0],
            [0.0, 0.0, -1.0, 8.0 + 1.0j],
        ],
        dtype=np.complex128,
    )
    b = np.array([6.0 + 1.0j, 25.0 + 2.0j, -11.0 + 1.0j, 15.0 - 1.0j], dtype=np.complex128)

    x, info = thomas(a, b)
    assert info == 0

    residual = a @ x - b
    assert np.allclose(residual, 0, atol=1e-12)
