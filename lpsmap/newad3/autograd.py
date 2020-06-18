# TODO: handle additional potentials

import numpy as np
import torch

from lpsmap.ad3qp.factor_graph import PFactorGraph

from .api import FactorGraph, Xor, Budget


class LPSparseMAP(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, fg, eta_u, *eta_v):
        fg.set_log_potentials(eta_u.detach().numpy())
        # perhaps to deal with eta_v:
        ctx.fg = fg
        ctx.shape = eta_u.shape
        value, u, add, status = fg.solve_qp_ad3()
        return torch.tensor(u, dtype=eta_u.dtype, device=eta_u.device)


    @classmethod
    def backward(cls, ctx, du):
        dtype = du.dtype
        device = du.device

        du = du.to(dtype=torch.double, device="cpu").detach().numpy()
        out = torch.empty(ctx.shape, dtype=torch.double, device='cpu')
        add = ctx.fg.jacobian_vec(du, out.numpy())

        # convert additional gradients into tensors
        add = (torch.tensor(x, dtype=dtype, device=device) for x in add if x)

        return None, out.to(dtype=dtype, device=device), *add


class TorchFactorGraph(FactorGraph):
    def variable(self, shape):
        scores = torch.zeros(shape)
        return self.variable_from(scores)

    def _cat(self, scores):
        return torch.cat(scores)

    def _ravel(self, x):
        return x.view(-1)

    def solve(self):
        pfg = PFactorGraph()

        offset, pvars, scores = self._make_variables(pfg)
        scores_add = self._make_factors(pfg, offset, pvars)
        print(scores_add)

        u = LPSparseMAP.apply(pfg, scores, *scores_add)

        for var in self.variables:
            k = offset[var]
            var.value = u[k:k + var._ix.size].reshape(var._ix.shape)


def main():
    fg = TorchFactorGraph()
    d = 4
    x = torch.randn(d, d, requires_grad=True)
    u = fg.variable_from(x)  # x are automatically used as scores

    for i in range(d):
        fg.add(Xor(u[i, :]))
        fg.add(Budget(u[:, i], budget=2))

    fg.solve()
    print(u.value)

    u.value[0, 0].backward()
    print(x.grad)


if __name__ == '__main__':
    main()

