# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool

from ..ad3qp.base cimport Factor, GenericFactor, PGenericFactor


cdef class PFactorTree(PGenericFactor):

    def __cinit__(self, bool allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorTreeTurbo()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    cpdef initialize(self, bool projective, bool packed, int length):

        cdef vector[pair[int, int]] arcs
        cdef int h, m

        if packed:
            for m in range(1, length + 1):
                for h in range(1, length + 1):
                    if h == m:
                        arcs.push_back(pair[int, int](0, m))
                    else:
                        arcs.push_back(pair[int, int](h, m))

        else:
            for m in range(1, length + 1):
                for h in range(length + 1):
                    if h != m:
                        arcs.push_back(pair[int, int](h, m))

        (<FactorTreeTurbo*>self.thisptr).Initialize(projective,
                                                    length + 1,
                                                    arcs)
