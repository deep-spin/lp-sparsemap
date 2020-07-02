"""Test basic API functionality"""
import numpy as np
import pytest

from lpsmap import FactorGraph, Xor


@pytest.mark.parametrize('shape', ((4,), (2, 2), (2, 2, 2)))
def test_no_factors(shape):

    rng = np.random.RandomState(32)
    x = rng.randn(*shape)
    fg = FactorGraph()
    u = fg.variable_from(x)
    fg.solve()

    expected = np.clip(x, 0, 1)
    obtained = u.value

    assert np.allclose(expected, obtained)


def test_slice_value():
    rng = np.random.RandomState(32)
    x = rng.randn(3, 3, 3)
    fg = FactorGraph()
    u = fg.variable_from(x)
    fg.solve()

    assert np.allclose(u[:].value, u.value[:])
    assert np.allclose(u[1].value, u.value[1])
    assert np.allclose(u[:, 1, :].value, u.value[:, 1, :])
    assert np.allclose(u[1, :, 0::2].value, u.value[1, :, 0::2])


def test_xor_satisfaction_overlap():
    rng = np.random.RandomState(32)
    x = rng.randn(3, 3)

    fg = FactorGraph()
    u = fg.variable_from(x)
    for i in range(3):
        fg.add(Xor(u[i, :]))
        fg.add(Xor(u[:, i]))
    fg.solve()

    assert np.allclose(u.value.sum(axis=0), 1)
    assert np.allclose(u.value.sum(axis=1), 1)

