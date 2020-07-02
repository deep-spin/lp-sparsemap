import torch
import pytest

from lpsmap import TorchFactorGraph, Xor

@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_smoke_data_types(dtype):

    x = torch.randn(5, dtype=dtype)
    fg = TorchFactorGraph()
    u = fg.variable_from(x)
    fg.solve()
    assert u.value.dtype == dtype


def test_grad():
    d = 3

    def f(x, v):
        fg = TorchFactorGraph()
        u = fg.variable_from(x)
        for i in range(d):
            fg.add(Xor(u[i, :]))
            fg.add(Xor(u[:, i]))
        fg.solve()

        # multiply by random vector
        # so that the backward pass is a directional gradient
        return (u.value * v).sum()

    torch.manual_seed(30)
    for _ in range(10):
        x = torch.randn(d, d, dtype=torch.double, requires_grad=True)
        v = torch.randn(d, d, dtype=torch.double, requires_grad=False)
        v /= torch.norm(v)
        torch.autograd.gradcheck(f, (x, v), atol=1e-3)
