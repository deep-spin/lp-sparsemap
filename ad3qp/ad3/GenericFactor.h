// Copyright (c) 2012 Andre Martins
// All Rights Reserved.
//
// This file is part of AD3 2.1.
//
// AD3 2.1 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// AD3 2.1 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with AD3 2.1.  If not, see <http://www.gnu.org/licenses/>.

#ifndef GENERIC_FACTOR_H_
#define GENERIC_FACTOR_H_

#include "Factor.h"

namespace AD3 {

// This must be implemented by the user-defined factor.
typedef void* Configuration;

// Base class for a generic factor.
// Specialized factors should be derived from this class.
class GenericFactor : public Factor
{
  public:
    GenericFactor()
    {
        verbosity_ = 2;
        num_max_iterations_QP_ = 10;
        clear_cache_ = true;
    }

    // Note: every class that derives from GenericFactor must
    // call ClearActiveSet() in their destructor.
    // We cannot call ClearActiveSet here because that will trigger
    // a call to the virtual function DeleteConfiguration.
    virtual ~GenericFactor(){};

    virtual int type() override { return FactorTypes::FACTOR_GENERIC; }
    bool IsGeneric() override { return true; }
    void SetVerbosity(int verbosity) { verbosity_ = verbosity; }

    /* Functions needed for gradient computation */
    void SetQPMaxIter(int it) { num_max_iterations_QP_ = it; }
    void SetClearCache(bool val) { clear_cache_ = val; }
    vector<Configuration> GetQPActiveSet() const { return active_set_; }
    vector<double> GetQPDistribution() const { return distribution_; }
    vector<double> GetQPInvA() const { return inverse_A_; }

    /* jacobian-vector wrt active set scores/probas */
    virtual void QVec(const double* v, double* out);

    /* jacobian-vector wrt posteriors */
    virtual void JacobianVec(const vector<double>& v,
                             vector<double>& out,
                             vector<double>& out_add) override;

    virtual void DistJacobianVec(const vector<double>& v,
                                 vector<double>& out,
                                 vector<double>& out_add);

    /* Get the correspondence between configurations & variable/additionals */
    void GetCorrespondence(vector<double>* M, vector<double>* N);

    virtual void SolveQP(const vector<double>& variable_log_potentials,
                         const vector<double>& additional_log_potentials,
                         vector<double>* variable_posteriors,
                         vector<double>* additional_posteriors) override;

  protected:
    void ClearActiveSet();

    // Compute posterior marginals from a sparse distribution,
    // expressed as a set of configurations (active_set) and
    // a probability/weight for each configuration (stored in
    // vector distribution).
    void ComputeMarginalsFromSparseDistribution(
      const vector<Configuration>& active_set,
      const vector<double>& distribution,
      vector<double>* variable_posteriors,
      vector<double>* additional_posteriors)
    {
        variable_posteriors->assign(binary_variables_.size(), 0.0);
        additional_posteriors->assign(additional_log_potentials_.size(), 0.0);
        for (size_t i = 0; i < active_set.size(); ++i) {
            UpdateMarginalsFromConfiguration(active_set[i],
                                             distribution[i],
                                             variable_posteriors,
                                             additional_posteriors);
        }
    }

    bool InvertAfterInsertion(const vector<Configuration>& active_set,
                              const Configuration& inserted_element);

    void InvertAfterRemoval(const vector<Configuration>& active_set,
                            int removed_index);

    void ComputeActiveSetSimilarities(const vector<Configuration>& active_set,
                                      vector<double>* similarities);

    void EigenDecompose(vector<double>* similarities,
                        vector<double>* eigenvalues);

    void Invert(const vector<double>& eigenvalues,
                const vector<double>& eigenvectors);

    bool IsSingular(vector<double>& eigenvalues,
                    vector<double>& eigenvectors,
                    vector<double>* null_space_basis);

  public:
    // Compute the score of a given assignment.
    // This must be implemented in the user-defined factor.
    virtual void Evaluate(const vector<double>& variable_log_potentials,
                          const vector<double>& additional_log_potentials,
                          const Configuration configuration,
                          double* value) = 0;

    // Find the most likely assignment.
    // This must be implemented in the user-defined factor.
    virtual void Maximize(const vector<double>& variable_log_potentials,
                          const vector<double>& additional_log_potentials,
                          Configuration& configuration,
                          double* value) = 0;

    // Given a configuration with a probability (weight),
    // increment the vectors of variable and additional posteriors.
    virtual void UpdateMarginalsFromConfiguration(
      const Configuration& configuration,
      double weight,
      vector<double>* variable_posteriors,
      vector<double>* additional_posteriors) = 0;

    // Count how many common values two configurations have.
    virtual int CountCommonValues(const Configuration& configuration1,
                                  const Configuration& configuration2) = 0;

    /* Generic implementation, ignores local optimizations. Can be overridden.
     */
    virtual double CountCommonValuesAdjusted(const Configuration& cfg1,
                                             const Configuration& cfg2)
    {
        auto mu_u1 = vector<double>(binary_variables_.size(), 0.0);
        auto mu_u2 = vector<double>(binary_variables_.size(), 0.0);
        auto mu_v = vector<double>(additional_log_potentials_.size(), 0.0);
        UpdateMarginalsFromConfiguration(cfg1, 1.0, &mu_u1, &mu_v);
        UpdateMarginalsFromConfiguration(cfg2, 1.0, &mu_u2, &mu_v);

        double ans = 0;
        for (unsigned int j = 0; j < binary_variables_.size(); ++j)
            ans += (mu_u1[j] * mu_u2[j]) / degrees_[j];

        return ans;
    }

    // chooses based on adjust_degrees which impl to use
    double CountCommonValuesAdapt(const Configuration& cfg1,
                                  const Configuration& cfg2)
    {
        if (adjust_degrees_)
            return CountCommonValuesAdjusted(cfg1, cfg2);
        else
            return CountCommonValues(cfg1, cfg2);
    }

    // Check if two configurations are the same.
    virtual bool SameConfiguration(const Configuration& configuration1,
                                   const Configuration& configuration2) = 0;

    // Create configuration.
    virtual Configuration CreateConfiguration() = 0;

    // Delete configuration.
    virtual void DeleteConfiguration(Configuration configuration) = 0;

    // Compute the MAP (local subproblem in the projected subgradient
    // algorithm). The user-defined factor may override this.
    virtual void SolveMAP(const vector<double>& variable_log_potentials,
                          const vector<double>& additional_log_potentials,
                          vector<double>* variable_posteriors,
                          vector<double>* additional_posteriors,
                          double* value)
    {
        Configuration configuration = CreateConfiguration();
        Maximize(variable_log_potentials,
                 additional_log_potentials,
                 configuration,
                 value);
        variable_posteriors->assign(variable_log_potentials.size(), 0.0);
        additional_posteriors->assign(additional_log_potentials.size(), 0.0);
        UpdateMarginalsFromConfiguration(
          configuration, 1.0, variable_posteriors, additional_posteriors);
        DeleteConfiguration(configuration);
    }

    void PrintActiveSet(std::ostream& out) {
        out << active_set_.size() << " ";
        for (auto i = 0u; i < active_set_.size(); ++i) {
            out << distribution_[i] << " ";
            PrintConfiguration(out, active_set_[i]);
            out << "\t";
        }
    }

    virtual void PrintConfiguration(std::ostream& out, const Configuration y) {
        auto mu_u = vector<double>(binary_variables_.size(), 0.0);
        auto mu_v = vector<double>(additional_log_potentials_.size(), 0.0);
        UpdateMarginalsFromConfiguration(y, 1.0, &mu_u, &mu_v);
        for (auto && ui : mu_u)
            out << ui << " ";
    }
    virtual void InitActiveSet(Configuration configuration);
    virtual void InitActiveSetFromScores(
        const vector<double>& variable_log_potentials,
        const vector<double>& additional_log_potentials);

  protected:
    vector<Configuration> active_set_;
    vector<double> distribution_;
    vector<double> inverse_A_;
    int num_max_iterations_QP_; // Initialize to 10.
    int verbosity_;             // Verbosity level.
    bool clear_cache_;
    Configuration init_configuration_ = nullptr;
};

} // namespace AD3

#endif // GENERIC_FACTOR_H_
