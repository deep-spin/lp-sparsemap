from lpsmap.ad3ext.sequence import PFactorSequence
from lpsmap.ad3ext.tree import PFactorTree

class Sequence(object):
    def __init__(self, variables, additionals):
        """

        Parameters
        ----------
        variables: shape (seq_length, n_states)

        additionals: shape (seq_length - 1, n_states, n_states)
            (transition scores)
        """
        self._variables = variables
        self._additionals = additionals

    def _construct(self, fg, pvars):
        seq_length, n_states = self._variables.shape
        f = PFactorSequence()
        f.initialize([n_states for _ in range(seq_length)])
        fg.declare_factor(f, pvars, owned_by_graph=True)

        return [f], [self._additionals]


class DepTree(object):
    def __init__(self, variables, packed=False, projective=False):
        """

        Parameters
        ----------
        variables: shape (seq_length, seq_length)

        projective, bool,
            whether to apply projectivity constraints.

        packed: bool,
            Controls the binary variable layout. If true, the diagonal elements
            u[i, i] denote the arc from word i to the root. If false,
            the first row (TODO CHECK) denote root arcs, while all
            subsequent rows skip over the self node.
        """
        self._variables = variables
        self.projective = projective
        self.packed = packed

    def _construct(self, fg, pvars):
        seq_length, _ = self._variables.shape
        f = PFactorTree()
        f.initialize(self.projective, self.packed, seq_length)
        fg.declare_factor(f, pvars, owned_by_graph=True)

        return [f], []
