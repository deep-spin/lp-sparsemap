# cython: language_level=2
# distutils: language=c++
# distutils: include_dirs = ad3qp

from cython cimport floating
from libcpp.vector cimport vector
from libcpp cimport bool


# get the classes from the c++ headers

cdef extern from "<iostream>" namespace "std":

    cdef cppclass ostream:
       ostream& write(const char*, int) except +
    ostream cout

cdef extern from "ad3/Factor.h" namespace "AD3":
    cdef cppclass BinaryVariable:
        BinaryVariable()
        double GetLogPotential()
        void SetLogPotential(double log_potential)
        int GetId()
        int Degree()

    cdef cppclass Factor:
        Factor()
        vector[double] GetAdditionalLogPotentials()
        void SetAdditionalLogPotentials(vector[double] additional_log_potentials)
        int Degree()
        int GetNumAdditionals()
        int GetLinkId(int i)
        bool IsGeneric()
        BinaryVariable* GetVariable(int i)
        void SolveMAP(vector[double] variable_log_potentials,
                      vector[double] additional_log_potentials,
                      vector[double] *variable_posteriors,
                      vector[double] *additional_posteriors,
                      double *value)
        void SolveQP(vector[double] variable_log_potentials,
                     vector[double] additional_log_potentials,
                     vector[double] *variable_posteriors,
                     vector[double] *additional_posteriors)
        void SolveQPAdjusted(
            vector[double] variable_log_potentials,
            vector[double] additional_log_potentials,
            vector[double] degrees,
            vector[double] *variable_posteriors,
            vector[double] *additional_posteriors)

        void JacobianVec(const vector[double] v, vector[double] out, vector[double] out_v);


cdef extern from "ad3/GenericFactor.h" namespace "AD3":
    ctypedef void *Configuration

    cdef cppclass GenericFactor(Factor):
        vector[Configuration] GetQPActiveSet()
        vector[double] GetQPDistribution()
        vector[double] GetQPInvA()
        void GetCorrespondence(vector[double]*, vector[double]*)
        void SetQPMaxIter(int)
        void SetClearCache(bool)
        void QVec(const double* v, double* out)
        void DistJacobianVec(const vector[double] v, vector[double] out, vector[double] out_v);
        void InitActiveSet(Configuration)
        void InitActiveSetFromScores(vector[double] variable_scores,
                                     vector[double] additional_scores)


cdef extern from "ad3/MultiVariable.h" namespace "AD3":
    cdef cppclass MultiVariable:
        int GetNumStates()
        BinaryVariable *GetState(int i)
        double GetLogPotential(int i)
        void SetLogPotential(int i, double log_potential)


cdef extern from "ad3/FactorGraph.h" namespace "AD3":
    cdef cppclass FactorGraph:
        FactorGraph()
        void SetVerbosity(int verbosity)
        void SetEtaPSDD(double eta)
        void SetMaxIterationsPSDD(int max_iterations)
        int SolveLPMAPWithPSDD(vector[double]* posteriors,
                               vector[double]* additional_posteriors,
                               double* value)
        void SetEtaAD3(double eta)
        void AdaptEtaAD3(bool adapt)
        void SetMaxIterationsAD3(int max_iterations)
        void SetResidualThresholdAD3(double threshold)
        void SetAutodetectAcyclic(bool autodetect)
        void FixMultiVariablesWithoutFactors()
        int SolveLPMAPWithAD3(vector[double]* posteriors,
                              vector[double]* additional_posteriors,
                              double* value)
        int SolveExactMAPWithAD3(vector[double]* posteriors,
                                 vector[double]* additional_posteriors,
                                 double* value)
        int SolveQP(vector[double]* posteriors,
                    vector[double]* additional_posteriors,
                    double* value)

        vector[vector[double]] JacobianVec(const double* du,
                                           double* out,
                                           unsigned int n_iter,
                                           double atol)

        vector[double] GetDualVariables()
        vector[double] GetLocalPrimalVariables()
        vector[double] GetGlobalPrimalVariables()

        BinaryVariable *CreateBinaryVariable()
        MultiVariable *CreateMultiVariable(int num_states)
        Factor *CreateFactorDense(vector[MultiVariable*] multi_variables,
                                  vector[double] additional_log_potentials,
                                  bool owned_by_graph)
        Factor *CreateFactorXOR(vector[BinaryVariable*] variables,
                                vector[bool] negated,
                                bool owned_by_graph)
        Factor *CreateFactorXOROUT(vector[BinaryVariable*] variables,
                                   vector[bool] negated,
                                   bool owned_by_graph)
        Factor *CreateFactorAtMostOne(vector[BinaryVariable*] variables,
                                      vector[bool] negated,
                                      bool owned_by_graph)
        Factor *CreateFactorOR(vector[BinaryVariable*] variables,
                               vector[bool] negated,
                               bool owned_by_graph)
        Factor *CreateFactorOROUT(vector[BinaryVariable*] variables,
                                  vector[bool] negated,
                                  bool owned_by_graph)
        Factor *CreateFactorANDOUT(vector[BinaryVariable*] variables,
                                   vector[bool] negated,
                                   bool owned_by_graph)
        Factor *CreateFactorIMPLY(vector[BinaryVariable*] variables,
                                  vector[bool] negated,
                                  bool owned_by_graph)
        Factor *CreateFactorPAIR(vector[BinaryVariable*] variables,
                                 double edge_log_potential,
                                 bool owned_by_graph)
        Factor *CreateFactorBUDGET(vector[BinaryVariable*] variables,
                                   vector[bool] negated,
                                   int budget,
                                   bool owned_by_graph)
        Factor *CreateFactorKNAPSACK(vector[BinaryVariable*] variables,
                                     vector[bool] negated,
                                     vector[double] costs,
                                     double budget,
                                     bool owned_by_graph)
        void DeclareFactor(Factor *factor,
                           vector[BinaryVariable*] variables,
                           bool owned_by_graph)

        void Print(ostream o)

        size_t GetNumVariables()
        size_t GetNumFactors()
        BinaryVariable* GetBinaryVariable(int)
        Factor* GetFactor(int)


# and the fundamental extension types

cdef class PBinaryVariable:
    cdef BinaryVariable *thisptr
    cdef bool allocate


cdef class PMultiVariable:
    cdef MultiVariable *thisptr
    cdef bool allocate

    cdef int _get_n_states(self)


cdef class PFactor:
    cdef Factor* thisptr
    cdef bool allocate


cdef class PGenericFactor(PFactor):
    cdef _cast_configuration(self, Configuration)
