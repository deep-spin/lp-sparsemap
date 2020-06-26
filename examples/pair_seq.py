import torch
from lpsmap import TorchFactorGraph, Pair

def main():

    torch.set_printoptions(precision=2, sci_mode=False)
    torch.manual_seed(4)

    n = 6

    x = torch.randn(n, requires_grad=True)
    transition = torch.randn(n - 1, requires_grad=True)

    fg = TorchFactorGraph()
    u = fg.variable_from(.1 * x)
    fg.add(Pair(u[:-1], u[1:], transition))
    fg.solve()

    print("solution:")
    print(u.value)

    print("gradients:")
    u.value[1].backward()
    print(x.grad)
    print(transition.grad)


if __name__ == '__main__':
    main()
