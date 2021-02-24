import torch
from lpsmap.ad3qp.factor_graph import PFactorGraph
from lpsmap import TorchFactorGraph, Pair, Budget
from lpsmap import Sequence, SequenceBudget
from lpsmap.ad3ext.sequence import PFactorSequence
import pdb

def main():
    torch.set_printoptions(precision=5, sci_mode=False)
    torch.manual_seed(1)
    temperature = 1.
    n = 12
    x = torch.randn((n,2), requires_grad=True)
    transition = torch.zeros((n+1,2,2), requires_grad=True)
    transition[:, 1, 1] = 1./temperature

    for budget in range(12):
        fg = TorchFactorGraph()
        u = fg.variable_from(x/temperature)

        fg.add(SequenceBudget(u, transition, budget))
        #fg.add(Sequence(u, transition))
        #fg.add(Budget(u, 2))
        fg.solve()

        print("solution: \n", u.value[:,0])
        print(sum(u.value[:,0]))

if __name__ == '__main__':
    main()
