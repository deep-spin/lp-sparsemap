# cython: language_level=2
# distutils: language=c++
# distutils: include_dirs = ad3qp

from libcpp.vector cimport vector
from libcpp cimport bool

from .base cimport FactorGraph

cdef class PFactorGraph:
    cdef FactorGraph *thisptr
