import numpy as np
from lpsmap.ad3qp.factor_graph import PFactorGraph

from .factors import Xor, Budget


class Variable(object):
    """AD3 binary variables packed as a tensor."""

    def __init__(self, scores):
        self.shape = scores.shape

        self._offset = None
        self._ix = np.arange(np.prod(self.shape)).reshape(self.shape)
        self._scores = scores

    def __getitem__(self, slice_arg):
        return Slice(self, slice_arg)

    def __repr__(self):
        return f"Variable(shape={self.shape})"

    # todo: operator~,  same gist as above
    # todo: how to
    # todo: smarter attr.value for redirection


class Slice(object):
    def __init__(self, var, slice_arg):
        self._base_var = var
        self._ix = self._base_var._ix[slice_arg]

    def __getitem__(self, slice_arg):
        return Slice(self._base_var, self._ix[slice_arg])

    @property
    def shape(self):
        return self._ix.shape

    @property
    def value(self):
        return self._base_var.value[self._ix]

    def __repr__(self):
        return f"Slice(shape={self.shape})"


class FactorGraph(object):

    def __init__(self):
        self.variables = []
        self.factors = []

    # todo: gradients here
    def variable_from(self, scores):
        var = Variable(scores=scores)
        self.variables.append(var)
        return var

    def variable(self, shape):
        scores = np.zeros(shape)
        return self.variable_from(scores)

    def add(self, factor):
        self.factors.append(factor)

    def _make_variables(self, pfg):
        offset = {}
        offset_ = 0

        pvars = []
        scores = []

        for var in self.variables:
            offset[var] = offset_
            offset_ += var._ix.size

            scores.append(var._scores.ravel())
            for i in range(var._ix.size):
                v = pfg.create_binary_variable()
                # v.set_log_potential(var._scores.flat[i])
                pvars.append(v)

        n_vars = offset_
        return offset, pvars, np.concatenate(scores)

    def solve(self):
        pfg = PFactorGraph()

        offset, pvars, scores = self._make_variables(pfg)

        for v, s in zip(pvars, scores):
            v.set_log_potential(s)

        pvars = np.array(pvars)  # so we may index by list

        for factor in self.factors:
            var = factor._variables
            if isinstance(var, Variable):
                ix = var._ix + offset[var]
                my_pvars = pvars[ix.ravel()].tolist()

            elif isinstance(var, Slice):
                ix = var._ix + offset[var._base_var]
                my_pvars = pvars[ix.ravel()].tolist()

            else:
                raise NotImplementedError()
            factor._construct(pfg, my_pvars)

        value, u, add, status = pfg.solve_qp_ad3()
        u = np.array(u)

        for var in self.variables:
            k = offset[var]
            var.value = u[k:k + var._ix.size].reshape(var._ix.shape)


def main():

    np.set_printoptions(precision=3, suppress=True)

    fg = FactorGraph()

    d = 4
    x = np.random.randn(d, d)
    u = fg.variable_from(x)  # x are automatically used as scores

    for i in range(d):
        fg.add(Xor(u[i, :]))
        fg.add(Budget(u[:, i], budget=2))

    fg.solve()
    print(u.value)


if __name__ == '__main__':
    main()

