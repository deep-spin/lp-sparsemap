import numpy as np
from lpsmap.ad3qp.factor_graph import PFactorGraph


class Variable(object):
    """AD3 binary variables packed as a tensor."""

    def __init__(self, scores):
        self.shape = scores.shape

        # self._fg = fg
        self._offset = None
        self._ix = np.arange(np.prod(self.shape)).reshape(self.shape)
        self._scores = scores

    def __getitem__(self, slice_arg):
        return Slice(self, slice_arg)

    def __repr__(self):
        return f"Variable(shape={self.shape}, ix={self._ix})"


    # todo: operator~,  same gist as above
    # todo: smarter attr.value for redirection


class Slice(object):
    def __init__(self, var, slice_arg):
        self._base_var = var
        # self._fg = self._base_var._fg
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
        return f"Slice(shape={self.shape}, ix={self._ix})"


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
        for var in self.variables:
            offset[var] = offset_
            offset_ += var._ix.size

            for i in range(var._ix.size):
                v = pfg.create_binary_variable()
                v.set_log_potential(var._scores[i])
                pvars.append(v)

        n_vars = offset_
        return offset, pvars


    def solve(self):
        pfg = PFactorGraph()
        offset, pvars = self._make_variables(pfg)

        pvars = np.array(pvars)  # so we may index by list

        for factor in self.factors:

            var = factor._variables
            if isinstance(var, Variable):
                ix = var._ix + offset[var]
                my_pvars = pvars[ix].tolist()

            elif isinstance(var, Slice):
                ix = var._ix + offset[var._base_var]
                my_pvars = pvars[ix].tolist()

            else:
                raise NotImplementedError()

            factor._construct(pfg, my_pvars)

        value, u, add, status = pfg.solve_qp_ad3()
        u = np.array(u)

        for var in self.variables:
            k = offset[var]
            var.value = u[k:k + var._ix.size].reshape(var._ix.shape)
        # print(posteriors)


# factors

class XOR(object):
    def __init__(self, variables):
        self._variables = variables

    # todo: deal with negated
    def _construct(self, fg, variables):
        return fg.create_factor_logic('XOR', variables)


def main():

    np.set_printoptions(precision=3, suppress=True)

    fg = FactorGraph()

    x = np.random.randn(6)
    u = fg.variable_from(x)

    # u_left = u[:4]
    u_left = u[:][:][:4]
    u_right = u[-4:]
    print(u, u_left, u_right)

    fg.add(XOR(u_left))
    fg.add(XOR(u_right))
    fg.solve()
    print(u.value)
    print(u_left.value)
    print(u_right.value)


if __name__ == '__main__':
    main()

