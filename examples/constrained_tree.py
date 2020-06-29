import torch
from lpsmap import TorchFactorGraph, DepTree, Budget


def main(n=5, constrain=False):

    print(f"n={n}, constrain={constrain}")

    torch.set_printoptions(precision=2, sci_mode=False)
    torch.manual_seed(4)

    x = torch.randn(n, n, requires_grad=True)

    fg = TorchFactorGraph()
    u = fg.variable_from(x)
    fg.add(DepTree(u, packed=True, projective=True))

    if constrain:
        for k in range(n):

            # don't constrain the diagonal (root arc)
            ix = list(range(k)) + list(range(k + 1, n))

            fg.add(Budget(u[ix, k], budget=2))

    fg.solve()
    print(u.value)

    u.value[1, -1].backward()
    print(x.grad)


if __name__ == '__main__':
    main(constrain=False)
    main(constrain=True)
