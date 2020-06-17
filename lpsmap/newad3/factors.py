# TODO: support axis=k to create multiple factors for each row/col
# TODO: knapsack (how to pass costs? must check broadcast/shape)

class Logic(object):
    def __init__(self, variables):
        self._variables = variables

    # TODO: deal with negated
    def _construct(self, fg, variables):
        return fg.create_factor_logic(self.factor_type, variables)


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
    def __init__(self, variables, budget):
        self._variables = variables
        self.budget = budget

    # TODO: deal with negated
    def _construct(self, fg, pvars):
        return fg.create_factor_budget(pvars, self.budget)
