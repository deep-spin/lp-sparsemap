import numpy as np
from numdifftools import Jacobian
from .test_logic_factors import _make_factor

fn = 'or'
fn = 'orout'
# fn = 'xor'
# fn = 'atmostone'
# fn = 'budget'
# fn = 'knap'

def check(eta, negated):

    d = eta.shape[0]

    # deg = .15 * np.ones(d)
    # deg = np.arange(1, d + 1, dtype=np.double)
    deg = np.ones(d)
    costs = np.random.RandomState(0).uniform(0, 3, size=d)
    g, f = _make_factor(eta, negated, fn, B=2, costs=costs)
    u, _ = f.solve_qp_adjusted(eta, [], degrees=deg)

    J_ = np.zeros((d, d))
    for j, v in enumerate(np.eye(d)):
        du, _ = f.jacobian_vec(v)
        J_[j] = np.array(du)

    xor = lambda x : np.array(f.solve_qp(x, [])[0])
    print(u)
    J = Jacobian(xor, step=.000001)(eta)
    print(J)
    print(J_)
    print(np.linalg.norm(J - J_))


def main():

    np.set_printoptions(suppress=True, precision=3)
    rng = np.random.RandomState(1)
    for _ in range(3):
        # eta = .1 * rng.randn(5)
        eta = 1 * rng.randn(5)
        # eta = np.abs(eta)

        neg = np.full(5, False)
        neg[1] = True
        neg2 = np.full(5, False)
        neg2[-1] = True
        negs = ([], neg, neg2)

        for negated in negs:
            print(eta)
            print(negated)
            check(eta, negated)
            print()

if __name__ == '__main__':
    main()
