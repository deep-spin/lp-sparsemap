# TODO: support axis=k to create multiple factors for each row/col
# TODO: knapsack (how to pass costs? must check broadcast/shape)

class Logic(object):
    def __init__(self, variables):
        self._variables = variables

    # TODO: deal with negated
    def _construct(self, fg, variables):
        return [fg.create_factor_logic(self.factor_type, variables)], []


class Xor(Logic):
    factor_type = "XOR"


class Or(Logic):
    factor_type = "OR"


class AtMostOne(Logic):
    factor_type = "ATMOSTONE"


class Imply(Logic):
    factor_type = "IMPLY"


class XorOut(Logic):
    factor_type = "XOROUT"


class OrOut(Logic):
    factor_type = "OROUT"


class AndOut(Logic):
    factor_type = "ANDOUT"


class Budget(object):
    def __init__(self, variables, budget, force_budget=False):
        self._variables = variables
        self.budget = budget
        self.force_budget = force_budget

    # TODO: deal with negated
    def _construct(self, fg, pvars):
        return [fg.create_factor_budget(pvars, self.budget,
                                        self.force_budget)], []


class Pair(object):
    # TODO: possible to have it be faster?
    def __init__(self, vars_i, vars_j, additionals):
        self._variables = vars_i, vars_j
        self._additionals = additionals

    def _construct(self, fg, pvars):
        vars_i, vars_j = pvars
        n = len(vars_i)
        adds = self._additionals
        factors = [
            fg.create_factor_pair([
                vars_i[k],
                vars_j[k]],
                adds[k])
            for k in range(n)
        ]
        add_tensors = [adds[k] for k in range(n)]
        return factors, add_tensors
