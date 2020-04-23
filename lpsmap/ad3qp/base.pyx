# cython: language_level=3
# distutils: language=c++
# cython: boundscheck=False

cimport cython
from cython cimport floating

from libcpp.vector cimport vector
from libcpp cimport bool

from .base cimport Factor
from .base cimport BinaryVariable
from .base cimport MultiVariable
from .base cimport Configuration


cdef class PBinaryVariable:

    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new BinaryVariable()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def get_log_potential(self):
        return self.thisptr.GetLogPotential()

    def set_log_potential(self, double log_potential):
        self.thisptr.SetLogPotential(log_potential)

    def get_id(self):
        return self.thisptr.GetId()

    def get_degree(self):
        return self.thisptr.Degree()


cdef class PMultiVariable:

    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new MultiVariable()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    cdef int _get_n_states(self):
        return self.thisptr.GetNumStates()

    def __len__(self):
        return self._get_n_states()

    def get_state(self, int i, bool validate=True):

        if validate and not 0 <= i < self._get_n_states():
            raise IndexError("State {:d} is out of bounds.".format(i))

        cdef BinaryVariable *variable = self.thisptr.GetState(i)
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def __getitem__(self, int i):

        if not 0 <= i < self._get_n_states():
            raise IndexError("State {:d} is out of bounds.".format(i))

        return self.get_log_potential(i)

    def __setitem__(self, int i, double log_potential):
        if not 0 <= i < len(self):
            raise IndexError("State {:d} is out of bounds.".format(i))
        self.set_log_potential(i, log_potential)

    def get_log_potential(self, int i):
        return self.thisptr.GetLogPotential(i)

    def set_log_potential(self, int i, double log_potential):
        self.thisptr.SetLogPotential(i, log_potential)

    @cython.boundscheck(False)
    def set_log_potentials(self, double[:] log_potentials, bool validate=True):
        cdef Py_ssize_t n_states = self.thisptr.GetNumStates()
        cdef Py_ssize_t i

        if validate and len(log_potentials) != n_states:
            raise IndexError("Expected buffer of length {}".format(n_states))

        for i in range(n_states):
            self.thisptr.SetLogPotential(i, log_potentials[i])


cdef class PFactor:
    # This is a virtual class, so don't allocate/deallocate.
    def __cinit__(self):
        self.allocate = False
        pass

    def __dealloc__(self):
        pass

    def set_allocate(self, allocate):
        self.allocate = allocate

    def get_additional_log_potentials(self):
        return self.thisptr.GetAdditionalLogPotentials()

    def set_additional_log_potentials(self,
                                      vector[double] additional_log_potentials):
        self.thisptr.SetAdditionalLogPotentials(additional_log_potentials)

    def get_degree(self):
        return self.thisptr.Degree()

    def get_link_id(self, int i):
        return self.thisptr.GetLinkId(i)

    def get_variable(self, int i):
        cdef BinaryVariable *variable = self.thisptr.GetVariable(i)
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def jacobian_vec(self, v):
        cdef vector[double] out
        cdef vector[double] out_v
        self.thisptr.JacobianVec(v, out, out_v);
        return out, out_v


    def solve_map(self, vector[double] variable_log_potentials,
                  vector[double] additional_log_potentials):
        """Solve maximization (MAP) for a single factor.

        Parameters
        ----------

        variable_scores, vector
            Unary scores, corresponds to eta_U,f.

        additional_scores, vector
            Additional scores, corresponds to eta_V,f.

        Returns
        -------

        value, double
            Score achieved by maximum structure.

        posteriors: vector
            Unary posterior SparseMAP marginals (mu_U)

        additional_posteriors: vector
            Additional posterior SparseMAP marginals (mu_V).
        """

        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        self.thisptr.SolveMAP(variable_log_potentials,
                              additional_log_potentials,
                              &posteriors,
                              &additional_posteriors,
                              &value)

        return value, posteriors, additional_posteriors

    def solve_qp(
            self,
            vector[double] variable_scores,
            vector[double] additional_scores):
        """Solve SparseMAP for a single factor.

        Parameters
        ----------

        variable_scores, vector
            Unary scores, corresponds to eta_U,f.

        additional_scores, vector
            Additional scores, corresponds to eta_V,f.

        Returns
        -------

        posteriors: vector
            Unary posterior SparseMAP marginals (mu_U).

        additional_posteriors: vector
            Additional posterior SparseMAP marginals (mu_V).
        """

        cdef:
            vector[double] posteriors
            vector[double] additional_posteriors

        additional_posteriors.resize(additional_scores.size())

        self.thisptr.SolveQP(
            variable_scores,
            additional_scores,
            &posteriors,
            &additional_posteriors)

        return posteriors, additional_posteriors

    def solve_qp_adjusted(
            self,
            vector[double] variable_scores,
            vector[double] additional_scores,
            vector[double] degrees):

        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors

        self.thisptr.SolveQPAdjusted(
            variable_scores,
            additional_scores,
            degrees,
            &posteriors,
            &additional_posteriors)

        return posteriors, additional_posteriors


cdef class PGenericFactor(PFactor):
    """Factor which uses the active set algorithm to solve its QP."""

    cdef _cast_configuration(self, Configuration cfg):
        """Cast a configuration to a python object.

        By default, we assume configurations are vectors of int.
        This can be overridden in custom factors."""

        return (<vector[int]*> cfg)[0]

    def solve_qp(
            self,
            vector[double] variable_scores,
            vector[double] additional_scores,
            int max_iter=10):
        """Solve SparseMAP for a single factor.

        Parameters
        ----------

        variable_scores, vector
            Unary scores, corresponds to eta_U,f.

        additional_scores, vector
            Additional scores, corresponds to eta_V,f.

        max_iter, int
            Number of active set iterations.

        Returns
        -------

        posteriors: vector
            Unary posterior SparseMAP marginals (mu_U).

        additional_posteriors: vector
            Additional posterior SparseMAP marginals (mu_V).
        """

        cdef GenericFactor* gf = <GenericFactor*?> self.thisptr
        gf.SetQPMaxIter(max_iter)
        return super().solve_qp(variable_scores, additional_scores)

    def solve_qp_adjusted(
            self,
            vector[double] variable_scores,
            vector[double] additional_scores,
            vector[double] degrees,
            int max_iter=10):

        cdef GenericFactor* gf = <GenericFactor*?> self.thisptr
        gf.SetQPMaxIter(max_iter)
        return super().solve_qp_adjusted(variable_scores, additional_scores,
                                         degrees)

    def q_vec(self, double[:] v, double[:] out):
        """Compute Jv of p wrt x, ie, Qv"""
        cdef GenericFactor* gf = <GenericFactor*?> self.thisptr
        gf.QVec(&v[0], &out[0])

    def dist_jacobian_vec(self, floating[:] dp,
                          floating[:] out,
                          floating[:] out_additional):
        """Compute Jacobian-vector product wrt. the distribution p_f.

        This computes the backward pass of the sparse vector p_f
        returned (in sparse form) by `get_sparse_solution`.

        Parameters
        ----------
        dp : np.array
            Vector to multiply by the Jacobian. In a neural network setting,
            this would be the gradient of the loss w.r.t. the nonzero
            coordinates of p_f (must have the same length as the number of
            selected structures).

        out : np.array,
            Memory location where the result will be stored. Must have same
            shape as the degree of the factor. In a neural network setting, this
            would be the gradient of the loss w.r.t. the variables scores eta.

        out_additional, list of lists
            Memory location to store gradients w.r.t. the additional scores of
            this factor. In a neural network setting, out_additional would be
            the gradient of the loss wrt the factor additional scores (eta_f,V).
        """
        cdef:
            Factor* f = self.thisptr
            GenericFactor* gf = <GenericFactor*?> f
            size_t n_active = gf.GetQPActiveSet().size()
            size_t degree = f.Degree();
            vector[double] dp_vec = vector[double](n_active)
            vector[double] out_vec = vector[double](degree);
            vector[double] out_add_vec;

        assert(dp.shape[0] == n_active)
        assert(out.shape[0] == degree)

        for i in range(n_active):
            dp_vec[i] = dp[i]

        gf.DistJacobianVec(dp_vec, out_vec, out_add_vec);

        for i in range(degree):
            out[i] = out_vec[i]

        if out_additional is not None:
            for i in range(out_add_vec.size()):
                out_additional[i] = out_add_vec[i]

    def get_sparse_solution(self):
        """Retrieve selected factor configurations and their probability.

        Must be called after SparseMAP optimization (`solve_qp`) has finished.

        Returns
        -------

        active_set : list,
            List of representations of the selected configurations. (Converted
            from internal C++ representations by _cast_configuration, which must
            be overridden if the representation is anything but a vector of
            integers.)

        distribution : list,
            Probability values of the selected configurations. Non-negative and
            sum to 1.
        """
        cdef:
            vector[Configuration] active_set_c
            vector[double] distribution
            vector[double] inverse_system
            vector[double] M, N
            GenericFactor* gf = <GenericFactor*?> self.thisptr

        active_set_c = gf.GetQPActiveSet()
        distribution = gf.GetQPDistribution()
        active_set_py = [self._cast_configuration(x) for x in active_set_c]

        return active_set_py, distribution

    def init_active_set_from_scores(self, vector[double] eta_u,
                                    vector[double] eta_v):
        cdef GenericFactor* gf = <GenericFactor*?> self.thisptr
        gf.InitActiveSetFromScores(eta_u, eta_v)

    def set_qp_iter(self, int max_iter):
        cdef GenericFactor* gf = <GenericFactor*?> self.thisptr
        gf.SetQPMaxIter(max_iter)

