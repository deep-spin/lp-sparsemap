# cython: language_level=2
# distutils: language=c++

from ..ad3qp.base cimport Factor, GenericFactor, PGenericFactor
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "FactorTreeTurbo.h" namespace "AD3":

    cdef cppclass FactorTreeTurbo(Factor):
        FactorTreeTurbo()
        void Initialize(bool, int, vector[pair[int, int]])


cdef class PFactorTree(PGenericFactor):
    cpdef initialize(self, bool projective, bool packed, int length)
