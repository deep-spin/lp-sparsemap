import numpy as np

from lpsmap.ad3qp.factor_graph import PFactorGraph
from lpsmap.ad3ext.factor_sequence import PFactorSequence


def sequence_sparsemap(constrain=False):
    if constrain:
        print("Sequence with budget constraints (LP-SparseMAP)")
    else:
        print("Sequence without constraints (SparseMAP)")

    length = 5
    n_states = 3

    # generate random scores
    rng = np.random.RandomState(13)

    eta_u = rng.randn(length * n_states)  # unaries

    n_add = (length - 1) * n_states ** 2  # a matrix for each transition
    n_add += 2 * n_states  # plus initial and final scores
    eta_v = rng.randn(n_add)  # non-stationary transition scores

    fg = PFactorGraph()
    f = PFactorSequence()

    variables = [fg.create_binary_variable()
                 for _ in range(length)
                 for _ in range(n_states)]

    # set potentials for first half of variables, rest default to zero
    for var, val in zip(variables, eta_u):
        var.set_log_potential(val)

    f.initialize([n_states for _ in range(length)])
    fg.declare_factor(f, variables)
    f.set_additional_log_potentials(eta_v)

    if constrain:
        # add budget constraint for each state
        for state in range(n_states):
            vars_state = variables[state::n_states]
            fg.create_factor_budget(vars_state, budget=2)

    if not constrain:
        # since there is a single factor, a single outer iter is enough.
        fg.set_eta_ad3(0.0)
        fg.set_max_iterations_ad3(1)

    # solve SparseMAP
    value, mu, mu_v, status = fg.solve_qp_ad3()
    mu = np.array(mu).reshape(length, n_states)
    print("mu:\n", mu, "\n\n")

    # perform backward pass
    du = rng.randn(length * n_states)
    out = np.empty_like(eta_u)
    fg.jacobian_vec(du, out)
    print("backward: ", out, "\n\n")


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    sequence_sparsemap(constrain=False)
    sequence_sparsemap(constrain=True)

