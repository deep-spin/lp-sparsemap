import torch
from lpsmap import TorchFactorGraph, Xor, AtMostOne

def main():

    torch.set_printoptions(precision=2, sci_mode=False)
    torch.manual_seed(4)

    m, n = 3, 5

    x = torch.randn(m, n, requires_grad=True)

    fg = TorchFactorGraph()
    u = fg.variable_from(x)  # x are automatically used as scores

    for i in range(m):
        fg.add(Xor(u[i, :]))

    for j in range(n):
        fg.add(AtMostOne(u[:, j]))  # some cols may be 0

    fg.solve()
    print(u.value)

    u.value[0, -1].backward()
    print(x.grad)


if __name__ == '__main__':
    main()

