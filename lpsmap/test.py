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
    #x.data[:, 0] += 5
    transition = torch.zeros((n+1,2,2), requires_grad=True)
    transition.data[1:n, 1, 1] = 1./temperature
    # Only one state in the beginning and in the end for start / stop symbol.
    transition = transition.reshape(-1)[2:-2]

    for budget in range(12):
        for force_budget in [True]:
            fg = TorchFactorGraph()
            u = fg.variable_from(x/temperature)

            fg.add(SequenceBudget(u, transition, budget, force_budget))
            fg.solve(verbose=0, autodetect_acyclic=True)
            
            print("Budget = %d:" % budget)
            
            print("solution: \n", u.value[:,0])
            print(sum(u.value[:,0]))


            #fg = TorchFactorGraph()
            #u = fg.variable_from(x/temperature)
            ## Sequence does not have transitions from/to start/end symbols.
            ## (Different from SequenceBudget.)
            #fg.add(Sequence(u, transition[2:-2]))
            #fg.add(Budget(u[:,0], budget, force_budget))
            #fg.solve(verbose=0, autodetect_acyclic=True)
            
            #print("Budget = %d:" % budget)
            
            #print("solution: \n", u.value[:,:])
            #print(sum(u.value[:,0]))


if __name__ == '__main__':
    main()
