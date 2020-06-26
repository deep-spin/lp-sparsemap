import torch
from lpsmap import TorchFactorGraph, DepTree, Budget

# from each row, select all but diagonal with a mask
def mask(k, n):
    m = torch.ones(n, dtype=torch.bool)
    m[k] = 0
    return m


def main(n=5, constrain=False):

    print(f"n={n}, constrain={constrain}")

    torch.manual_seed(4)
    x = torch.randn(n, n, requires_grad=True)

    fg = TorchFactorGraph()
    u = fg.variable_from(x)
    fg.add(DepTree(u, packed=True, projective=True))

    if constrain:
        for k in range(n):
            fg.add(Budget(u[mask(k, n), k], budget=2))

    fg.solve()
    print(u.value)

    u.value[1, -1].backward()
    print(x.grad)


if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False)
    main(constrain=False)
    main(constrain=True)
