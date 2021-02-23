# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool
from libcpp.vector cimport vector

from ..ad3qp.base cimport PGenericFactor

cdef class PFactorSequenceBudget(PGenericFactor):

    def __cinit__(self, bool allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequenceBudget()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] num_states, int budget):
        (<FactorSequenceBudget*>self.thisptr).Initialize(num_states, budget)
