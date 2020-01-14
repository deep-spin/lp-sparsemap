import numpy as np

# make edge case log-potential inputs:
def make_edge_cases(d):

    edge_cases = []

    zero = np.zeros(d, dtype=np.double)
    bige = np.full(d, +100, dtype=np.double)
    smol = np.full(d, -100, dtype=np.double)

    for base in (zero, bige, smol):
        edge_cases.append(base)
        for i in range(d):
            v = base.copy()
            v[i] = 1
            edge_cases.append(v)

            v = base.copy()
            v[i] = 100
            edge_cases.append(v)

            v = base.copy()
            v[i] = -1
            edge_cases.append(v)

            v = base.copy()
            v[i] = -100
            edge_cases.append(v)

    return edge_cases
