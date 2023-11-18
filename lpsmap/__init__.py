from .version import __version__
from .api.factors import (Xor, Or, AtMostOne, Imply, XorOut, OrOut,
                         AndOut, Budget, Pair)
from .api.factors_extension import Sequence, DepTree, SequenceBudget
from .api.api import FactorGraph

try:
    from .api.autograd import TorchFactorGraph
except ImportError:
    pass

