import pytest
import torch
from lpsmap import TorchFactorGraph, DepTree, Pair


@pytest.mark.parametrize("projective", (False, True))
def test_tree_map(projective):
    """test that MAP on a single factor graph is exact"""

    torch.manual_seed(1)

    for _ in range(20):

        x = torch.randn(4, 4)
        fg = TorchFactorGraph()
        u = fg.variable_from(x)
        fg.add(DepTree(u, packed=True, projective=projective))
        fg.solve_map(autodetect_acyclic=False, max_iter=100)
        u_iter = u.value

        fg = TorchFactorGraph()
        u = fg.variable_from(x)
        fg.add(DepTree(u, packed=True, projective=projective))
        fg.solve_map(autodetect_acyclic=True, max_iter=1)

        u_exact = u.value

        assert torch.all(torch.unique(u_exact) == torch.tensor([0, 1]))
        assert torch.allclose(u_exact, u_iter)

def test_map_with_additionals():

    # factor graph x1-x2 x3-x4 (two independent pairs)

    x = torch.randn(4)
    t = torch.randn(2)

    fg = TorchFactorGraph()
    u = fg.variable_from(x)
    fg.add(Pair(u[0], u[1], t[0:1]))
    fg.add(Pair(u[2], u[3], t[1:2]))
    fg.solve_map(autodetect_acyclic=True)
    u_exact = u.value

    assert torch.all(torch.unique(u_exact) == torch.tensor([0, 1]))

