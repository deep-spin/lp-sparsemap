try:
    import torch
except ImportError as e:
    import warnings
    warnings.warn("pytorch cannot be found: TorchFactorGraph not available.")
    raise e


import numpy as np

from lpsmap.ad3qp.factor_graph import PFactorGraph

from .api import FactorGraph


class _LPSparseMAP(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, fg, eta_u, *eta_v):
        fg.set_log_potentials(eta_u.detach().numpy())
        detached_eta_v = [np.atleast_1d(x.detach().numpy()) for x in eta_v]
        fg.set_all_additionals(detached_eta_v)
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

        return (None, out.to(dtype=dtype, device=device), *add)


class TorchFactorGraph(FactorGraph):
    def variable(self, shape):
        scores = torch.zeros(shape)
        return self.variable_from(scores)

    def _cat(self, scores):
        return torch.cat(scores)

    def _ravel(self, x):
        return x.view(-1)

    def solve(self, max_iter=1000, max_inner_iter=10, tol=1e-6, step_size=.1,
            adapt=True, verbose=False):
        """
         Parameters
        ---------
        max_iter : int, default: 1000
            Maximum number of iterations to perform.

        max_inner_iter: int, default: 10
            Maximum per-factor Active Set iterations

        tol : double, default: 1e-6
            Theshold for the primal and dual residuals in AD3. The algorithm
            ends early when both residuals are below this threshold.

        step_size : float, default: 0.1
            Value of the penalty constant. If adapt_eta is true, this is the
            initial penalty, otherwise every iteration will apply this amount
            of penalty.

        adapt : boolean, default: True
            If true, adapt the penalty constant using the strategy in [2].

        verbose : int, optional
            Degree of verbosity of debugging information to display. By default,
            nothing is printed.

        """
        pfg = PFactorGraph()

        offset, pvars, scores = self._make_variables(pfg)
        scores_add = self._make_factors(pfg, offset, pvars)

        pfg.set_verbosity(verbose)
        pfg.set_eta_ad3(step_size)
        pfg.adapt_eta_ad3(adapt)
        pfg.set_max_iterations_ad3(max_iter)
        pfg.set_residual_threshold_ad3(tol)
        pfg.set_inner_iter(max_inner_iter)

        u = _LPSparseMAP.apply(pfg, scores, *scores_add)

        for var in self.variables:
            k = offset[var]
            var.value = u[k:k + var._ix.size].reshape(var._ix.shape)


    def solve_map(self, max_iter=1000, max_inner_iter=10, tol=1e-6, step_size=.1,
            adapt=True, verbose=False):
        """
         Parameters
        ---------
        max_iter : int, default: 1000
            Maximum number of iterations to perform.

        max_inner_iter: int, default: 10
            Maximum per-factor Active Set iterations

        tol : double, default: 1e-6
            Theshold for the primal and dual residuals in AD3. The algorithm
            ends early when both residuals are below this threshold.

        step_size : float, default: 0.1
            Value of the penalty constant. If adapt_eta is true, this is the
            initial penalty, otherwise every iteration will apply this amount
            of penalty.

        adapt : boolean, default: True
            If true, adapt the penalty constant using the strategy in [2].

        verbose : int, optional
            Degree of verbosity of debugging information to display. By default,
            nothing is printed.

        """
        pfg = PFactorGraph()

        offset, pvars, scores = self._make_variables(pfg)
        scores_add = self._make_factors(pfg, offset, pvars)

        pfg.set_verbosity(100)
        pfg.set_eta_ad3(step_size)
        pfg.adapt_eta_ad3(adapt)
        pfg.set_max_iterations_ad3(max_iter)
        pfg.set_residual_threshold_ad3(tol)
        pfg.set_inner_iter(max_inner_iter)

        eta_u = scores
        eta_v = scores_add
        pfg.set_log_potentials(eta_u.detach().numpy())
        detached_eta_v = [np.atleast_1d(x.detach().numpy()) for x in eta_v]
        pfg.set_all_additionals(detached_eta_v)
        value, u, add, status = pfg.solve_lp_map_ad3()
        print(status)
        u = torch.tensor(u, dtype=eta_u.dtype, device=eta_u.device)

        for var in self.variables:
            k = offset[var]
            var.value = u[k:k + var._ix.size].reshape(var._ix.shape)

