# Check entire logic factor zoo

import itertools
import numpy as np
import pytest

from numpy.testing import assert_allclose

from lpsmap.ad3qp.factor_graph import PFactorGraph
from lpsmap.sparsemap_fw import SparseMAPFW

from .make_edge_cases import make_edge_cases


def _func_best_structure(eta, negated, f):

    d = eta.shape[0]

    # generate all possible assignments
    scores = []
    cfgs = []
    for u in itertools.product(*([0, 1] for _ in range(d))):
        u = np.array(u)
        u_neg = u.copy()
        u_neg[negated] = 1 - u_neg[negated]

        if not f(u_neg):
            continue
        score = np.dot(u, eta)
        scores.append(score)
        cfgs.append(u)

    ix = np.argmax(scores)
    score = scores[ix]
    u = cfgs[ix]
    return score, u


def _make_factor(eta, negated, fname, **kwargs):
    negated_flag = np.zeros_like(eta, dtype=np.bool)
    negated_flag[negated] = True
    negated_flag = list(negated_flag)

    g = PFactorGraph()
    variables = []
    for x in eta:
        v = g.create_binary_variable()
        v.set_log_potential(x)
        variables.append(v)

    if fname == 'budget':
        f = g.create_factor_budget(variables, kwargs['B'], negated_flag)

    elif fname == 'knap':
        f = g.create_factor_knapsack(variables, kwargs['costs'], kwargs['B'], negated_flag)

    else:
        f = g.create_factor_logic(fname.upper(), variables, negated_flag)

    return g, f


class FunctionalPolytope(object):
    def __init__(self, negated, f, deg=None):
        self.negated = negated
        self.f = f
        if deg is not None:
            self.q = np.sqrt(deg)
        else:
            self.q = 1

    def vertex(self, y):
        u = np.array(y, dtype=np.double)
        u /= self.q
        return u, np.array(())

    def map_oracle(self, eta_u, eta_v):
        eta = eta_u / self.q
        val, y = _func_best_structure(eta, self.negated, self.f)
        return tuple(y)


d = 5
rng = np.random.RandomState(0)
B = 3

FUNCS = {
    'or':  lambda u : np.any(u == 1),
    'xor': lambda u : np.sum(u == 1) == 1,
    'atmostone': lambda u : np.sum(u == 1) <= 1,
    'budget': lambda u : np.sum(u == 1) <= B,
    'orout': lambda u : np.any(u[:-1] == 1) == u[-1],  # tricky
}


TEST_POINTS = make_edge_cases(d)
for _ in range(10):
    TEST_POINTS.append(rng.randn(d))

tol = dict(rtol=.1, atol=1e-8)

@pytest.mark.parametrize('negated', ([], [1, 2], [-1]))
@pytest.mark.parametrize('fname', FUNCS.keys())
@pytest.mark.parametrize('eta', TEST_POINTS)
def test_map(negated, fname, eta):
    f = FUNCS[fname]
    score_exp, u_exp = _func_best_structure(eta, negated, f)

    g, f = _make_factor(eta, negated, fname, B=B)
    score_got, u_got, _ = f.solve_map(eta, [])
    assert_allclose(score_exp, score_got, **tol)
    # not necessarily unique!
    # assert_allclose(u_exp, u_got)


@pytest.mark.parametrize('negated', ([], [1, 2], [-1]))
@pytest.mark.parametrize('fname', FUNCS.keys())
@pytest.mark.parametrize('eta', TEST_POINTS)
def test_qp(negated, fname, eta):
    if fname == 'orout':
        pytest.skip('suspicious behaviour, irrelevant for LPSparseMAP for now')

    f = FUNCS[fname]
    fp = FunctionalPolytope(negated, f)  # B / costs baked into f
    fw = SparseMAPFW(fp, max_iter=1000, tol=1e-12)
    u_exp, _, _ = fw.solve(eta, [])
    g, f = _make_factor(eta, negated, fname, B=B)
    u_got, _ = f.solve_qp(eta, [])
    assert_allclose(u_exp, u_got, **tol)


@pytest.mark.parametrize('negated', ([], [1, 2], [-1]))
@pytest.mark.parametrize('fname', FUNCS.keys())
@pytest.mark.parametrize('eta', TEST_POINTS)
def test_qp_adjusted(negated, fname, eta):
    deg = np.arange(1, d + 1, dtype=np.double)
    f = FUNCS[fname]
    fp = FunctionalPolytope(negated, f, deg=deg)
    fw = SparseMAPFW(fp, max_iter=1000, tol=1e-12)
    u_exp, _, _ = fw.solve(eta, [])
    g, f = _make_factor(eta, negated, fname, B=B)
    u_got, _ = f.solve_qp_adjusted(eta, [], degrees=deg)
    assert_allclose(u_exp, u_got, **tol)


@pytest.mark.parametrize('negated', ([], [1, 2], [-1]))
@pytest.mark.parametrize('fname', FUNCS.keys())
@pytest.mark.parametrize('eta', TEST_POINTS)
def test_qp_adj_equiv(negated, fname, eta):
    if fname == 'orout':
        pytest.skip('suspicious behaviour, irrelevant for LPSparseMAP for now')
    deg = np.ones(d)
    f = FUNCS[fname]
    g, f = _make_factor(eta, negated, fname, B=B)
    u_exp, _ = f.solve_qp(eta, [])
    u_got, _ = f.solve_qp_adjusted(eta, [], degrees=deg)
    assert_allclose(u_exp, u_got, **tol)
