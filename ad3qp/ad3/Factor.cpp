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

#include "Factor.h"
#include "Utils.h"

namespace AD3 {

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int
FactorXOR::AddEvidence(vector<bool>* active_links,
                       vector<int>* evidence,
                       vector<int>* additional_evidence)
{
    bool changes = false;

    // Look for absorbing elements.
    size_t k;
    for (k = 0; k < Degree(); ++k) {
        if (!(*active_links)[k])
            continue;
        if ((*evidence)[k] < 0)
            continue;
        if ((!negated_[k] && (*evidence)[k] == 1) ||
            (negated_[k] && (*evidence)[k] == 0)) {
            break;
        }
    }
    if (k < Degree()) {
        // Found absorbing element. Set evidence to all the other inputs and
        // disable the factor.
        for (size_t l = 0; l < Degree(); ++l) {
            (*active_links)[l] = false;
            if (k == l)
                continue;
            int value = negated_[l] ? 1 : 0;
            // If evidence was set otherwise for this input, return
            // contradiction.
            if ((*evidence)[l] >= 0 && (*evidence)[l] != value)
                return -1;
            (*evidence)[l] = value;
        }
        // Return code to disable factor.
        return 2;
    }

    // Look for neutral elements.
    int num_active = 0;
    for (k = 0; k < Degree(); ++k) {
        if (!(*active_links)[k])
            continue;
        ++num_active;
        if ((*evidence)[k] < 0)
            continue;
        if ((!negated_[k] && (*evidence)[k] == 0) ||
            (negated_[k] && (*evidence)[k] == 1)) {
            // Neutral element found. Make it inactive and proceed.
            (*active_links)[k] = false;
            --num_active;
            changes = true;
        }
    }
    // If there are no active variables, return contradiction.
    if (num_active == 0)
        return -1;
    // If there is only one active variable, set evidence to that variable
    // and disable the factor.
    if (num_active == 1) {
        for (k = 0; k < Degree(); ++k) {
            if ((*active_links)[k])
                break;
        }
        assert(k < Degree());
        (*active_links)[k] = false;
        int value = negated_[k] ? 0 : 1;
        // If evidence was set otherwise for this input, return contradiction.
        if ((*evidence)[k] >= 0 && (*evidence)[k] != value)
            return -1;
        (*evidence)[k] = value;
        return 2;
    }

    return changes ? 1 : 0;
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void
FactorXOR::SolveMAP(const vector<double>& variable_log_potentials,
                    const vector<double>& additional_log_potentials,
                    vector<double>* variable_posteriors,
                    vector<double>* additional_posteriors,
                    double* value)
{
    variable_posteriors->resize(variable_log_potentials.size());

    // Create a local copy of the log potentials.
    vector<double> log_potentials(variable_log_potentials);

    int first = -1;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            log_potentials[f] = -log_potentials[f];
    }

    *value = 0.0;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            *value -= log_potentials[f];
    }

    for (size_t f = 0; f < Degree(); ++f) {
        if (first < 0 || log_potentials[f] > log_potentials[first])
            first = f;
    }

    *value += log_potentials[first];
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f]) {
            (*variable_posteriors)[f] = 1.0;
        } else {
            (*variable_posteriors)[f] = 0.0;
        }
    }
    (*variable_posteriors)[first] = negated_[first] ? 0.0 : 1.0;
}

// Solve the QP (local subproblem in the AD3 algorithm).
void
FactorXOR::SolveQP(const vector<double>& variable_log_potentials,
                   const vector<double>& additional_log_potentials,
                   vector<double>* variable_posteriors,
                   vector<double>* additional_posteriors)
{
    size_t d = Degree();
    variable_posteriors->resize(d);

    FlipNegatedForQP(variable_log_potentials, variable_posteriors);

    if (adjust_degrees_) {
        project_onto_weighted_simplex(
          *variable_posteriors, degrees_, sqrt_degrees_);
    } else {
        project_onto_simplex_cached(variable_posteriors->data(),
                                    /*size=*/d,
                                    /*sum_to=*/1,
                                    last_sort_);
    }

    for (size_t j = 0; j < d; ++j)
        support_[j] = (*variable_posteriors)[j] > 0;

    FlipNegatedForQP(*variable_posteriors, variable_posteriors);
}

void
FactorXOR::JacobianVec(const vector<double>& v,
                       vector<double>& out,
                       vector<double>& out_add)
{
    size_t d = Degree();
    out.assign(d, 0);

    for (size_t j = 0; j < d; ++j)
        if (support_[j])
            out[j] = v[j];

    FlipSignsInPlace(out);

    double sum_v = 0, ssz = 0;

    for (size_t j = 0; j < d; ++j) {
        if (support_[j]) {
            if (adjust_degrees_) {
                ssz += degrees_[j];
                sum_v += sqrt_degrees_[j] * out[j];
            } else {
                ssz += 1;
                sum_v += out[j];
            }
        }
    }
    sum_v /= ssz;

    for (size_t j = 0; j < d; ++j)
        if (support_[j])
            out[j] -= (adjust_degrees_ ? sqrt_degrees_[j] * sum_v : sum_v);

    FlipSignsInPlace(out);
}

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int
FactorAtMostOne::AddEvidence(vector<bool>* active_links,
                             vector<int>* evidence,
                             vector<int>* additional_evidence)
{
    bool changes = false;

    // Look for absorbing elements.
    size_t k;
    for (k = 0; k < Degree(); ++k) {
        if (!(*active_links)[k])
            continue;
        if ((*evidence)[k] < 0)
            continue;
        if ((!negated_[k] && (*evidence)[k] == 1) ||
            (negated_[k] && (*evidence)[k] == 0)) {
            break;
        }
    }
    if (k < Degree()) {
        // Found absorbing element. Set evidence to all the other inputs and
        // disable the factor.
        for (size_t l = 0; l < Degree(); ++l) {
            (*active_links)[l] = false;
            if (k == l)
                continue;
            int value = negated_[l] ? 1 : 0;
            // If evidence was set otherwise for this input, return
            // contradiction.
            if ((*evidence)[l] >= 0 && (*evidence)[l] != value)
                return -1;
            (*evidence)[l] = value;
        }
        // Return code to disable factor.
        return 2;
    }

    // Look for neutral elements.
    int num_active = 0;
    for (k = 0; k < Degree(); ++k) {
        if (!(*active_links)[k])
            continue;
        ++num_active;
        if ((*evidence)[k] < 0)
            continue;
        if ((!negated_[k] && (*evidence)[k] == 0) ||
            (negated_[k] && (*evidence)[k] == 1)) {
            // Neutral element found. Make it inactive and proceed.
            (*active_links)[k] = false;
            --num_active;
            changes = true;
        }
    }
    // If there are no active variables, disable the factor.
    if (num_active == 0)
        return 2;
    // If there is only one active variable, disable that link
    // and disable the factor.
    if (num_active == 1) {
        for (k = 0; k < Degree(); ++k) {
            if ((*active_links)[k])
                break;
        }
        assert(k < Degree());
        (*active_links)[k] = false;
        return 2;
    }

    return changes ? 1 : 0;
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void
FactorAtMostOne::SolveMAP(const vector<double>& variable_log_potentials,
                          const vector<double>& additional_log_potentials,
                          vector<double>* variable_posteriors,
                          vector<double>* additional_posteriors,
                          double* value)
{
    variable_posteriors->resize(variable_log_potentials.size());

    // Create a local copy of the log potentials.
    vector<double> log_potentials(variable_log_potentials);

    int first = -1;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            log_potentials[f] = -log_potentials[f];
    }

    *value = 0.0;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            *value -= log_potentials[f];
    }

    for (size_t f = 0; f < Degree(); ++f) {
        if (first < 0 || log_potentials[f] > log_potentials[first])
            first = f;
    }

    bool all_zeros = true;
    if (log_potentials[first] > 0.0) {
        *value += log_potentials[first];
        all_zeros = false;
    }

    for (size_t f = 0; f < Degree(); f++) {
        if (negated_[f]) {
            (*variable_posteriors)[f] = 1.0;
        } else {
            (*variable_posteriors)[f] = 0.0;
        }
    }

    if (!all_zeros)
        (*variable_posteriors)[first] = negated_[first] ? 0.0 : 1.0;
}

// Solve the QP (local subproblem in the AD3 algorithm).
void
FactorAtMostOne::SolveQP(const vector<double>& variable_log_potentials,
                         const vector<double>& additional_log_potentials,
                         vector<double>* variable_posteriors,
                         vector<double>* additional_posteriors)
{
    size_t d = Degree();
    variable_posteriors->resize(d);

    FlipNegatedForQP(variable_log_potentials, variable_posteriors);

    // clip to [0, 1] or [0, 1 / sqrt(d_j)]
    double sum = Clip(variable_posteriors);
    tight_ = false;

    if (sum > 1) {
        tight_ = true;

        // clipping does not yield a feasible solution.
        // The sum constraint must be tight sum == 1.

        // first, set the posteriors back to original
        FlipNegatedForQP(variable_log_potentials, variable_posteriors);

        if (adjust_degrees_) {
            project_onto_weighted_simplex(
              *variable_posteriors, degrees_, sqrt_degrees_);
        } else {
            project_onto_simplex_cached(variable_posteriors->data(),
                                        /*size=*/d,
                                        /*sum_to=*/1,
                                        last_sort_);
        }

        // save support for backward pass
        // NOTE: technically, we should check the upper bound here too
        // but because of XOR, there is at most one variable at its UB,
        // and it automatically gets 0 in backward pass.
        for (size_t j = 0; j < d; ++j)
            support_[j] = (*variable_posteriors)[j] > 0;
    }

    FlipNegatedForQP(*variable_posteriors, variable_posteriors);
}

void
FactorAtMostOne::JacobianVec(const vector<double>& v,
                             vector<double>& out,
                             vector<double>& out_add)
{
    size_t d = Degree();
    out.assign(d, 0);

    for (size_t j = 0; j < d; ++j)
        if (support_[j])
            out[j] = v[j];

    FlipSignsInPlace(out);

    if (tight_) {
        // we had to project to simplex, so
        double sum_v = 0, ssz = 0;

        for (size_t j = 0; j < d; ++j) {
            if (support_[j]) {
                if (adjust_degrees_) {
                    ssz += degrees_[j];
                    sum_v += sqrt_degrees_[j] * out[j];
                } else {
                    ssz += 1;
                    sum_v += v[j];
                }
            }
        }
        sum_v /= ssz;

        for (size_t j = 0; j < d; ++j)
            if (support_[j])
                out[j] -= (adjust_degrees_ ? sqrt_degrees_[j] * sum_v : sum_v);
    }
    FlipSignsInPlace(out);
}

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int
FactorOR::AddEvidence(vector<bool>* active_links,
                      vector<int>* evidence,
                      vector<int>* additional_evidence)
{
    bool changes = false;

    // Look for absorbing elements.
    size_t k;
    for (k = 0; k < Degree(); ++k) {
        if (!(*active_links)[k])
            continue;
        if ((*evidence)[k] < 0)
            continue;
        if ((!negated_[k] && (*evidence)[k] == 1) ||
            (negated_[k] && (*evidence)[k] == 0)) {
            break;
        }
    }
    if (k < Degree()) {
        // Found absorbing element. Disable the factor and all links.
        for (size_t l = 0; l < Degree(); ++l) {
            if (!(*active_links)[l])
                continue;
            (*active_links)[l] = false;
        }
        // Return code to disable factor.
        return 2;
    }

    // Look for neutral elements.
    int num_active = 0;
    for (k = 0; k < Degree(); ++k) {
        if (!(*active_links)[k])
            continue;
        ++num_active;
        if ((*evidence)[k] < 0)
            continue;
        if ((!negated_[k] && (*evidence)[k] == 0) ||
            (negated_[k] && (*evidence)[k] == 1)) {
            // Neutral element found. Make it inactive and proceed.
            (*active_links)[k] = false;
            --num_active;
            changes = true;
        }
    }

    // If there are no active variables, return contradiction.
    if (num_active == 0)
        return -1;
    // If there is only one active variable, set evidence to that variable
    // and disable the factor.
    if (num_active == 1) {
        for (k = 0; k < Degree(); ++k) {
            if ((*active_links)[k])
                break;
        }
        assert(k < Degree());
        (*active_links)[k] = false;
        int value = negated_[k] ? 0 : 1;
        // If evidence was set otherwise for this input, return contradiction.
        if ((*evidence)[k] >= 0 && (*evidence)[k] != value)
            return -1;
        (*evidence)[k] = value;
        return 2;
    }

    return changes ? 1 : 0;
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void
FactorOR::SolveMAP(const vector<double>& variable_log_potentials,
                   const vector<double>& additional_log_potentials,
                   vector<double>* variable_posteriors,
                   vector<double>* additional_posteriors,
                   double* value)
{
    variable_posteriors->resize(variable_log_potentials.size());

    // Create a local copy of the log potentials.
    vector<double> log_potentials(variable_log_potentials);

    int first = -1;
    double valaux;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            log_potentials[f] = -log_potentials[f];
    }

    *value = 0.0;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            *value -= log_potentials[f];
    }

    for (size_t f = 0; f < Degree(); ++f) {
        valaux = log_potentials[f];
        if (valaux < 0.0) {
            valaux = 0.0;
            (*variable_posteriors)[f] = negated_[f] ? 1.0 : 0.0;
        } else {
            (*variable_posteriors)[f] = negated_[f] ? 0.0 : 1.0;
        }
        *value += valaux;
    }

    for (size_t f = 0; f < Degree(); ++f) {
        if (first < 0 || log_potentials[f] > log_potentials[first]) {
            first = f;
        }
    }
    valaux = log_potentials[first];
    // valaux = min(0,valaux);
    if (valaux > 0.0) {
        valaux = 0.0;
    } else {
        (*variable_posteriors)[first] = negated_[first] ? 0.0 : 1.0;
    }
    *value += valaux;
}

// Solve the QP (local subproblem in the AD3 algorithm).
void
FactorOR::SolveQP(const vector<double>& variable_log_potentials,
                  const vector<double>& additional_log_potentials,
                  vector<double>* variable_posteriors,
                  vector<double>* additional_posteriors)
{
    size_t d = Degree();
    variable_posteriors->resize(d);
    FlipNegatedForQP(variable_log_potentials, variable_posteriors);

    // clip to [0, 1] or [0, 1 / sqrt(d_j)], and check the sum constraint.
    double sum = Clip(variable_posteriors);
    tight_ = false;

    if (sum < 1) {
        tight_ = true;
        // clipping does not yield a feasible solution.
        // The sum constraint must be tight sum == 1.

        // first, set the posteriors back to original
        FlipNegatedForQP(variable_log_potentials, variable_posteriors);
        if (adjust_degrees_) {
            project_onto_weighted_simplex(
              *variable_posteriors, degrees_, sqrt_degrees_);
        } else {
            project_onto_simplex_cached(variable_posteriors->data(),
                                        /*size=*/d,
                                        /*sum_to=*/1,
                                        last_sort_);
        }

        // save support for backward pass
        // NOTE: technically, we should check the upper bound here too
        // but because of XOR, there is at most one variable at its UB,
        // and it automatically gets 0 in backward pass.
        for (size_t j = 0; j < d; ++j)
            support_[j] = (*variable_posteriors)[j] > 0;
    }

    FlipNegatedForQP(*variable_posteriors, variable_posteriors);
}

void
FactorOR::JacobianVec(const vector<double>& v,
                      vector<double>& out,
                      vector<double>& out_add)
{
    // identical code to FactorAtMostOne, but tight_ means opposite.
    size_t d = Degree();
    out.assign(d, 0);

    for (size_t j = 0; j < d; ++j)
        if (support_[j])
            out[j] = v[j];

    FlipSignsInPlace(out);

    if (tight_) {
        // we had to project to simplex, so
        double sum_v = 0, ssz = 0;

        for (size_t j = 0; j < d; ++j) {
            if (support_[j]) {
                if (adjust_degrees_) {
                    ssz += degrees_[j];
                    sum_v += sqrt_degrees_[j] * out[j];
                } else {
                    ssz += 1;
                    sum_v += out[j];
                }
            }
        }
        sum_v /= ssz;

        for (size_t j = 0; j < d; ++j)
            if (support_[j])
                out[j] -= (adjust_degrees_ ? sqrt_degrees_[j] * sum_v : sum_v);
    }
    FlipSignsInPlace(out);
}

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int
FactorOROUT::AddEvidence(vector<bool>* active_links,
                         vector<int>* evidence,
                         vector<int>* additional_evidence)
{
    bool changes = false;

    // 1) Look for absorbing elements in the first N-1 inputs.
    size_t k;
    for (k = 0; k < Degree() - 1; ++k) {
        if (!(*active_links)[k])
            continue;
        if ((*evidence)[k] < 0)
            continue;
        if ((!negated_[k] && (*evidence)[k] == 1) ||
            (negated_[k] && (*evidence)[k] == 0)) {
            break;
        }
    }
    if (k < Degree() - 1) {
        // Found absorbing element. Set evidence to the last input and
        // disable the factor.
        for (size_t l = 0; l < Degree(); ++l) {
            if (!(*active_links)[l])
                continue;
            (*active_links)[l] = false;
        }
        size_t l = Degree() - 1;
        int value = negated_[l] ? 0 : 1;
        // If evidence was set otherwise for this input, return contradiction.
        if ((*evidence)[l] >= 0 && (*evidence)[l] != value)
            return -1;
        (*evidence)[l] = value;

        // Return code to disable factor.
        return 2;
    }

    // 2) Look for neutral elements in the first N-1 inputs.
    int num_active = 0;
    for (k = 0; k < Degree() - 1; ++k) {
        if (!(*active_links)[k])
            continue;
        ++num_active;
        if ((*evidence)[k] < 0)
            continue;
        if ((!negated_[k] && (*evidence)[k] == 0) ||
            (negated_[k] && (*evidence)[k] == 1)) {
            // Neutral element found. Make it inactive and proceed.
            (*active_links)[k] = false;
            --num_active;
            changes = true;
        }
    }

    // If there are no active variables in the first N-1 inputs,
    // set evidence to the last variable and disable the factor.
    if (num_active == 0) {
        int l = Degree() - 1;
        (*active_links)[l] = false;
        int value = negated_[l] ? 1 : 0;
        // If evidence was set otherwise for this input, return contradiction.
        if ((*evidence)[l] >= 0 && (*evidence)[l] != value)
            return -1;
        (*evidence)[l] = value;
        return 2;
    }

    // 3) Handle the last input.
    k = Degree() - 1;
    if ((*active_links)[k] && (*evidence)[k] >= 0) {
        if ((!negated_[k] && (*evidence)[k] == 0) ||
            (negated_[k] && (*evidence)[k] == 1)) {
            // Absorbing element. Set evidence to all variables and disable the
            // factor.
            (*active_links)[k] = false;
            for (size_t l = 0; l < Degree() - 1; ++l) {
                if (!(*active_links)[l])
                    continue;
                (*active_links)[l] = false;
                int value = negated_[l] ? 1 : 0;
                // If evidence was set otherwise for this input, return
                // contradiction.
                if ((*evidence)[l] >= 0 && (*evidence)[l] != value)
                    return -1;
                (*evidence)[l] = value;
            }
            return 2;
        } else {
            // (!negated_[k] && evidence[k] == 1) ||
            // (negated_[k] && evidence[k] == 0))
            // For now, just disable the last link.
            // Later, turn the factor into a OR factor.
            (*active_links)[k] = false;
            changes = true;
        }
    }

    return changes ? 1 : 0;
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void
FactorOROUT::SolveMAP(const vector<double>& variable_log_potentials,
                      const vector<double>& additional_log_potentials,
                      vector<double>* variable_posteriors,
                      vector<double>* additional_posteriors,
                      double* value)
{
    variable_posteriors->resize(variable_log_potentials.size());

    // Create a local copy of the log potentials.
    vector<double> log_potentials(variable_log_potentials);

    int first = -1;
    double valaux;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            log_potentials[f] = -log_potentials[f];
    }

    for (size_t f = 0; f < Degree(); ++f) {
        (*variable_posteriors)[f] = 0.0;
    }

    for (size_t f = 0; f < Degree() - 1; ++f) {
        if (first < 0 || log_potentials[f] > log_potentials[first]) {
            first = f;
        }
    }
    valaux = log_potentials[first];
    // valaux = min(0,valaux);
    if (valaux > 0.0) {
        valaux = 0.0;
    } else {
        (*variable_posteriors)[first] = 1.0;
    }
    *value = valaux;

    for (size_t f = 0; f < Degree() - 1; ++f) {
        valaux = log_potentials[f];
        // valaux = max(0,valaux);
        if (valaux < 0.0) {
            valaux = 0.0;
        } else {
            (*variable_posteriors)[f] = 1.0;
        }
        *value += valaux;
    }

    *value += log_potentials[Degree() - 1];
    if (*value < 0.0) {
        *value = 0.0;
        for (size_t f = 0; f < Degree(); ++f) {
            (*variable_posteriors)[f] = 0.0;
        }
    } else {
        (*variable_posteriors)[Degree() - 1] = 1.0;
    }

    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f]) {
            *value -= log_potentials[f];
            (*variable_posteriors)[f] = 1 - (*variable_posteriors)[f];
            // log_potentials[f] = -log_potentials[f];
        }
    }
}

// Solve the QP (local subproblem in the AD3 algorithm).
void
FactorOROUT::SolveQP(const vector<double>& variable_log_potentials,
                     const vector<double>& additional_log_potentials,
                     vector<double>* variable_posteriors,
                     vector<double>* additional_posteriors)
{
    size_t d = Degree();
    variable_posteriors->resize(d);

    // 1) Start by projecting onto the cubed cone = conv (.*1, 0)
    // Project onto the unit cube
    //
    FlipNegatedForQP(variable_log_potentials, variable_posteriors);
    double sum = Clip(variable_posteriors);

    tight_cone_ = false;
    tight_sum_ = false;
    size_t f;

    // check if z in A1: z[k] <= Z[d] forall k < d?

    if (adjust_degrees_) {
        for (f = 0; f < d - 1; ++f)
            if ((*variable_posteriors)[f] * sqrt_degrees_[f] >
                (*variable_posteriors)[d - 1] * sqrt_degrees_[d - 1])
                break;
    } else {
        for (f = 0; f < d - 1; ++f)
            if ((*variable_posteriors)[f] > (*variable_posteriors)[d - 1])
                break;
    }

    if (f < d - 1) {
        // there exists k, z[k] > z[d], so z not in A1
        tight_cone_ = true;

        // Project onto cone A1: z[k] <= z[d] forall k
        FlipNegatedForQP(variable_log_potentials, variable_posteriors);

        tight_ineq_.assign(d, false);
        if (adjust_degrees_) {
            project_onto_weighted_cone(
              *variable_posteriors, degrees_, sqrt_degrees_, tight_ineq_);
        } else {
            project_onto_cone_cached(variable_posteriors->data(),
                                     /*size=*/d,
                                     last_sort_,
                                     tight_ineq_);
        }

        // Project onto the unit cube again
        sum = Clip(variable_posteriors);
        for (size_t j = 0; j < d; ++j)
            if (!support_[j])
                tight_ineq_[j] = false;
    }

    // 2) Add the inequality  sum_k<d z[k]  >= z[d]

    double last = (*variable_posteriors)[d - 1];
    if (adjust_degrees_)
        last *= sqrt_degrees_[d - 1];
    sum -= last;

    if (sum < last) {
        // Project onto xor with negated output.
        // Careful: temporarily alters negated_ until end of function.
        tight_sum_ = true;
        negated_[d - 1] = !negated_[d - 1];
        FlipNegatedForQP(variable_log_potentials, variable_posteriors);
        if (adjust_degrees_) {
            project_onto_weighted_simplex(
              *variable_posteriors, degrees_, sqrt_degrees_);
        } else {
            project_onto_simplex_cached(variable_posteriors->data(),
                                        /*size=*/d,
                                        /*sum_to=*/1,
                                        last_sort_);
        }

        for (size_t j = 0; j < d; ++j)
            support_[j] = (*variable_posteriors)[j] > 0;
    }

    FlipNegatedForQP(*variable_posteriors, variable_posteriors);

    if (tight_sum_) // leave negated_ as we found it
        negated_[d - 1] = !negated_[d - 1];
}

void
FactorOROUT::JacobianVec(const vector<double>& v,
                         vector<double>& out,
                         vector<double>& out_add)
{
    size_t d = Degree();

    // clip to box: this is common among all branches
    out.assign(d, 0);
    for (size_t j = 0; j < d; ++j)
        if (support_[j])
            out[j] = v[j];

    FlipSignsInPlace(out);

    // if tight_sum_: backward pass of XOR with output
    if (tight_sum_) {
        out[d - 1] *= -1; // flip sign

        double sum_v = 0, ssz = 0;

        for (size_t j = 0; j < d; ++j) {
            if (support_[j]) {
                if (adjust_degrees_) {
                    ssz += degrees_[j];
                    sum_v += sqrt_degrees_[j] * out[j];
                } else {
                    ssz += 1;
                    sum_v += out[j];
                }
            }
        }
        sum_v /= ssz;

        for (size_t j = 0; j < d; ++j)
            if (support_[j])
                out[j] -= (adjust_degrees_ ? sqrt_degrees_[j] * sum_v : sum_v);

        out[d - 1] *= -1; // flip sign back
    }
    // if tight_cone_, then we projected onto the (cubed) cone
    else if (tight_cone_) {
        double sum_v = 0, ssz = 0;

        for (size_t j = 0; j < d; ++j) {
            if (tight_ineq_[j]) {
                if (adjust_degrees_) {
                    ssz += 1 / degrees_[j];
                    sum_v += out[j] / sqrt_degrees_[j];
                } else {
                    ssz += 1;
                    sum_v += out[j];
                }
            }
        }
        sum_v /= ssz;

        // below is hard assignment, not in-place add
        for (size_t j = 0; j < d; ++j)
            if (tight_ineq_[j])
                out[j] = (adjust_degrees_ ? (sum_v / sqrt_degrees_[j]) : sum_v);
    }
    // else, if neither: we projected onto cube, this is already done at top.

    FlipSignsInPlace(out);
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void
FactorBUDGET::SolveMAP(const vector<double>& variable_log_potentials,
                       const vector<double>& additional_log_potentials,
                       vector<double>* variable_posteriors,
                       vector<double>* additional_posteriors,
                       double* value)
{
    variable_posteriors->resize(variable_log_potentials.size());

    // Create a local copy of the log potentials.
    vector<double> log_potentials(variable_log_potentials);

    double valaux;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            log_potentials[f] = -log_potentials[f];
    }

    *value = 0.0;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            *value -= log_potentials[f];
    }

    size_t num_active = 0;
    double sum = 0.0;
    for (size_t f = 0; f < Degree(); ++f) {
        valaux = log_potentials[f];
        if (valaux < 0.0) {
            (*variable_posteriors)[f] = negated_[f] ? 1.0 : 0.0;
        } else {
            sum += valaux;
            (*variable_posteriors)[f] = negated_[f] ? 0.0 : 1.0;
        }
        ++num_active;
    }

    if (num_active > GetBudget()) {
        vector<pair<double, int>> scores(Degree());
        for (size_t f = 0; f < Degree(); ++f) {
            scores[f].first = -log_potentials[f];
            scores[f].second = f;
        }
        sort(scores.begin(), scores.end());
        num_active = 0;
        sum = 0.0;
        for (size_t k = 0; k < GetBudget(); ++k) {
            valaux = -scores[k].first;
            if (valaux < 0.0 && !ForcedBudget())
                break;
            int f = scores[k].second;
            (*variable_posteriors)[f] = negated_[f] ? 0.0 : 1.0;
            sum += valaux;
            ++num_active;
        }
        for (size_t k = num_active; k < Degree(); ++k) {
            int f = scores[k].second;
            (*variable_posteriors)[f] = negated_[f] ? 1.0 : 0.0;
        }
    }

    *value += sum;
}

// Solve the QP (local subproblem in the AD3 algorithm).
void
FactorBUDGET::SolveQP(const vector<double>& variable_log_potentials,
                      const vector<double>& additional_log_potentials,
                      vector<double>* variable_posteriors,
                      vector<double>* additional_posteriors)
{
    size_t d = Degree();
    variable_posteriors->resize(d);
    FlipNegatedForQP(variable_log_potentials, variable_posteriors);

    // clip to [0, 1] or [0, 1 / sqrt(d_j)], and check the sum constraint.
    double s = Clip(variable_posteriors);
    double budget = GetBudget();
    tight_ = false;

    if (s > budget || ForcedBudget()) {
        tight_ = true;

        FlipNegatedForQP(variable_log_potentials, variable_posteriors);

        if (adjust_degrees_) {
            auto costs = vector<double>(d, 1.0); // reuse knapsack code
            // NOTE: if we add a budget param to weighted_simplex, can reuse
            project_onto_weighted_knapsack(
              *variable_posteriors, costs, budget, degrees_, sqrt_degrees_);
        } else {
            project_onto_budget_constraint_cached(&(*variable_posteriors)[0],
                                                  /*size=*/d,
                                                  budget,
                                                  last_sort_);
        }

        double val, UB;
        for (size_t j = 0; j < d; ++j) {
            val = (*variable_posteriors)[j];
            UB = adjust_degrees_ ? 1 / sqrt_degrees_[j] : 1;
            support_[j] = (val > 0 && val < UB); // seems to work without slack
        }
    }
    FlipNegatedForQP(*variable_posteriors, variable_posteriors);
}

void
FactorBUDGET::JacobianVec(const vector<double>& v,
                          vector<double>& out,
                          vector<double>& out_add)
{
    size_t d = Degree();
    out.assign(d, 0);

    for (size_t j = 0; j < d; ++j)
        if (support_[j])
            out[j] = v[j];

    FlipSignsInPlace(out);

    if (tight_) {
        // we had to project to (capped) simplex, so
        double sum_v = 0, ssz = 0;

        for (size_t j = 0; j < d; ++j) {
            if (support_[j]) {
                if (adjust_degrees_) {
                    ssz += degrees_[j];
                    sum_v += sqrt_degrees_[j] * out[j];
                } else {
                    ssz += 1;
                    sum_v += v[j];
                }
            }
        }
        sum_v /= ssz;

        for (size_t j = 0; j < d; ++j)
            if (support_[j])
                out[j] -= (adjust_degrees_ ? sqrt_degrees_[j] * sum_v : sum_v);
    }
    FlipSignsInPlace(out);
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void
FactorKNAPSACK::SolveMAP(const vector<double>& variable_log_potentials,
                         const vector<double>& additional_log_potentials,
                         vector<double>* variable_posteriors,
                         vector<double>* additional_posteriors,
                         double* value)
{
    /*
    cout << "Solve LP" << endl;
    for (int f = 0; f < Degree(); ++f) {
      cout << variable_log_potentials[f] << " ";
    }
    cout << endl;
    for (int f = 0; f < Degree(); ++f) {
      cout << costs_[f] << " ";
    }
    cout << endl;
    */

    variable_posteriors->resize(variable_log_potentials.size());

    // Create a local copy of the log potentials.
    vector<double> log_potentials(variable_log_potentials);

    double valaux;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            log_potentials[f] = -log_potentials[f];
    }

    *value = 0.0;
    for (size_t f = 0; f < Degree(); ++f) {
        if (negated_[f])
            *value -= log_potentials[f];
    }

    double total_cost = 0.0;
    double sum = 0.0;
    for (size_t f = 0; f < Degree(); ++f) {
        valaux = log_potentials[f];
        if (valaux < 0.0) {
            (*variable_posteriors)[f] = negated_[f] ? 1.0 : 0.0;
        } else {
            sum += valaux;
            (*variable_posteriors)[f] = negated_[f] ? 0.0 : 1.0;
        }
        total_cost += GetCost(f);
    }

    if (total_cost > GetBudget()) {
        vector<pair<double, int>> scores(Degree());
        for (size_t f = 0; f < Degree(); ++f) {
            scores[f].first = -log_potentials[f] / GetCost(f);
            scores[f].second = f;
        }
        sort(scores.begin(), scores.end());
        total_cost = 0.0;
        sum = 0.0;
        size_t num_active = 0;
        for (size_t k = 0; k < Degree(); ++k) {
            int f = scores[k].second;
            valaux = log_potentials[f];
            if (valaux < 0.0)
                break;
            if (total_cost + GetCost(f) > GetBudget()) {
                double posterior = (GetBudget() - total_cost) / GetCost(f);
                (*variable_posteriors)[f] =
                  negated_[f] ? 1.0 - posterior : posterior;
                sum += valaux * posterior;
                total_cost = GetBudget();
                ++num_active;
                break;
            }

            (*variable_posteriors)[f] = negated_[f] ? 0.0 : 1.0;
            sum += valaux;
            total_cost += GetCost(f);
            ++num_active;
        }
        for (size_t k = num_active; k < Degree(); ++k) {
            int f = scores[k].second;
            (*variable_posteriors)[f] = negated_[f] ? 1.0 : 0.0;
        }
    }

    *value += sum;

    // for (int f = 0; f < Degree(); ++f) {
    //  cout << (*variable_posteriors)[f] << " ";
    //}
    // cout << endl;
    // cout << *value << endl;
}

// Solve the QP (local subproblem in the AD3 algorithm).
void
FactorKNAPSACK::SolveQP(const vector<double>& variable_log_potentials,
                        const vector<double>& additional_log_potentials,
                        vector<double>* variable_posteriors,
                        vector<double>* additional_posteriors)
{
    size_t d = Degree();
    variable_posteriors->resize(d);
    FlipNegatedForQP(variable_log_potentials, variable_posteriors);

    // clip to [0, 1] or [0, 1 / sqrt(d_j)]
    Clip(variable_posteriors);
    double budget = GetBudget();

    double s = 0.0;
    for (size_t f = 0; f < d; ++f) {
        double val = GetCost(f) * (*variable_posteriors)[f];
        if (adjust_degrees_)
            val *= sqrt_degrees_[f];
        s += val;
    }

    tight_ = false;

    if (s > budget) {
        tight_ = true;
        FlipNegatedForQP(variable_log_potentials, variable_posteriors);

        if (adjust_degrees_) {
            project_onto_weighted_knapsack(
              *variable_posteriors, costs_, budget, degrees_, sqrt_degrees_);
        } else {
            project_onto_knapsack_constraint(&(*variable_posteriors)[0],
                                             &costs_[0],
                                             /*size=*/d,
                                             budget);
        }
        double val, UB;
        for (size_t j = 0; j < d; ++j) {
            val = (*variable_posteriors)[j];
            UB = adjust_degrees_ ? 1 / sqrt_degrees_[j] : 1;
            support_[j] = (val > 0 && val < UB); // seems to work without slack
        }
    }

    FlipNegatedForQP(*variable_posteriors, variable_posteriors);
}

void
FactorKNAPSACK::JacobianVec(const vector<double>& v,
                            vector<double>& out,
                            vector<double>& out_add)
{
    size_t d = Degree();
    out.assign(d, 0);

    for (size_t j = 0; j < d; ++j)
        if (support_[j])
            out[j] = v[j];

    FlipSignsInPlace(out);

    if (tight_) {
        // we had to project to (capped) simplex, so
        double sum_v = 0, ssz = 0;

        for (size_t j = 0; j < d; ++j) {
            if (support_[j]) {
                if (adjust_degrees_) {
                    ssz += degrees_[j] * costs_[j] * costs_[j];
                    sum_v += sqrt_degrees_[j] * costs_[j] * out[j];
                } else {
                    ssz += costs_[j] * costs_[j];
                    sum_v += costs_[j] * v[j];
                }
            }
        }
        sum_v /= ssz;

        for (size_t j = 0; j < d; ++j)
            if (support_[j])
                out[j] -=
                  (adjust_degrees_ ? sqrt_degrees_[j] * costs_[j] * sum_v
                                   : costs_[j] * sum_v);
    }
    FlipSignsInPlace(out);
}

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int
FactorPAIR::AddEvidence(vector<bool>* active_links,
                        vector<int>* evidence,
                        vector<int>* additional_evidence)
{
    bool changes = false;
    additional_evidence->assign(1, -1);

    // If there is no evidence, do nothing and return "no changes."
    if ((*evidence)[0] < 0 && (*evidence)[1] < 0)
        return 0;
    if ((*evidence)[0] >= 0 && (*evidence)[1] >= 0) {
        if ((*evidence)[0] == 1 && (*evidence)[1] == 1) {
            (*additional_evidence)[0] = 1;
        } else {
            (*additional_evidence)[0] = 0;
        }
        (*active_links)[0] = (*active_links)[1] = false;
        return 2;
    }
    // Only one of the variables has evidence. Disable all links and, depending
    // on the evidence, keep or discard the factor.
    if ((*active_links)[0] || (*active_links)[1]) {
        changes = true;
        (*active_links)[0] = false;
        (*active_links)[1] = false;
    }
    if ((*evidence)[0] >= 0) {
        if ((*evidence)[0] == 0) {
            (*additional_evidence)[0] = 0;
            return 2;
        } else {
            return changes ? 1 : 0;
        }
    } else { // (*evidence)[1] >= 0.
        if ((*evidence)[1] == 0) {
            (*additional_evidence)[0] = 0;
            return 2;
        } else {
            return changes ? 1 : 0;
        }
    }
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
// Remark: (*additional_posteriors)[0] will be 1 iff
// (*variable_posteriors)[0] = (*variable_posteriors)[1] = 1.
// Remark: assume inputs are NOT negated.
void
FactorPAIR::SolveMAP(const vector<double>& variable_log_potentials,
                     const vector<double>& additional_log_potentials,
                     vector<double>* variable_posteriors,
                     vector<double>* additional_posteriors,
                     double* value)
{
    variable_posteriors->resize(variable_log_potentials.size());
    additional_posteriors->resize(additional_log_potentials.size());

    double p[4] = {
        0.0,                        // 00
        variable_log_potentials[1], // 01
        variable_log_potentials[0], // 10
        variable_log_potentials[0] + variable_log_potentials[1] +
          additional_log_potentials[0] // 11
    };

    int best = 0;
    for (int i = 1; i < 4; i++) {
        if (p[i] > p[best])
            best = i;
    }

    *value = p[best];
    if (best == 0) {
        (*variable_posteriors)[0] = 0.0;
        (*variable_posteriors)[1] = 0.0;
        (*additional_posteriors)[0] = 0.0;
    } else if (best == 1) {
        (*variable_posteriors)[0] = 0.0;
        (*variable_posteriors)[1] = 1.0;
        (*additional_posteriors)[0] = 0.0;
    } else if (best == 2) {
        (*variable_posteriors)[0] = 1.0;
        (*variable_posteriors)[1] = 0.0;
        (*additional_posteriors)[0] = 0.0;
    } else { // if (best == 3)
        (*variable_posteriors)[0] = 1.0;
        (*variable_posteriors)[1] = 1.0;
        (*additional_posteriors)[0] = 1.0;
    }
}

// Solve the QP (local subproblem in the AD3 algorithm).
void
FactorPAIR::SolveQP(const vector<double>& variable_log_potentials,
                    const vector<double>& additional_log_potentials,
                    vector<double>* variable_posteriors,
                    vector<double>* additional_posteriors)
{
    variable_posteriors->resize(variable_log_potentials.size());
    additional_posteriors->resize(additional_log_potentials.size());

    // min 1/2 (u[0] - u0[0])^2 + (u[1] - u0[1])^2 + u0[2] * u[2],
    // where u[2] is the edge marginal.
    // Remark: Assume inputs are NOT negated.
    double x0[3] = { variable_log_potentials[0],
                     variable_log_potentials[1],
                     -additional_log_potentials[0] };

    double d0 = 1, d1 = 1;

    if (adjust_degrees_) {
        d0 = 1 / sqrt_degrees_[0];
        d1 = 1 / sqrt_degrees_[1];
    }

    double r = d1 / d0;

    double c = x0[2];
    flipped_ = false;
    if (additional_log_potentials[0] < 0) {
        flipped_ = true;
        x0[0] -= c / d0;
        x0[1] = d1 - x0[1];
        c = -c;
    }

    if ((d1 * x0[0]) > (d0 * x0[1]) - (c / r)) {
        branch_ = 0;
        (*variable_posteriors)[0] = x0[0];
        (*variable_posteriors)[1] = x0[1] - c / d1;
    } else if ((d0 * x0[1]) > (d1 * x0[0]) - (c * r)) {
        branch_ = 1;
        (*variable_posteriors)[0] = x0[0] - c / d0;
        (*variable_posteriors)[1] = x0[1];
    } else {
        branch_ = 2;
        (*variable_posteriors)[0] =
          (d0 * x0[0] + d1 * x0[1] - c) / (d0 + r * d1);
        (*variable_posteriors)[1] = r * (*variable_posteriors)[0];
    }

    // Project onto box.

    clip_u0_ = false;
    clip_u1_ = false;
    if ((*variable_posteriors)[0] < 0.0) {
        (*variable_posteriors)[0] = 0.0;
        clip_u0_ = true;
    } else if ((*variable_posteriors)[0] > d0) {
        (*variable_posteriors)[0] = d0;
        clip_u0_ = true;
    }
    if ((*variable_posteriors)[1] < 0.0) {
        (*variable_posteriors)[1] = 0.0;
        clip_u1_ = true;
    } else if ((*variable_posteriors)[1] > d1) {
        (*variable_posteriors)[1] = d1;
        clip_u1_ = true;
    }

    // u[2] = min(u[0] / d0, u[1] / d1);

    double u0d0 = (*variable_posteriors)[0] / d0;
    double u1d1 = (*variable_posteriors)[1] / d1;

    if (u0d0 <= u1d1) {
        (*additional_posteriors)[0] = u0d0;
        clip_u01_ = clip_u0_;
    } else {
        (*additional_posteriors)[0] = u1d1;
        clip_u01_ = clip_u1_;
    }

    //(*additional_posteriors)[0] =
    //((*variable_posteriors)[0] / d0 < (*variable_posteriors)[1] / d1)?
    //(*variable_posteriors)[0] / d0 : (*variable_posteriors)[1] / d1;

    if (flipped_) { // c > 0
        (*variable_posteriors)[1] = d1 - (*variable_posteriors)[1];
        (*additional_posteriors)[0] = u0d0 - (*additional_posteriors)[0];

        if (u0d0 <=
            u1d1 + 1e12) // tolerance factor needed when they are almost equal
            clip_u01_ = true; // always becomes 0
    }

    // clip_u01_ = (*additional_posteriors)[0] <= 1e-10 ||
    // (*additional_posteriors)[0] >= 1;
}

void
FactorPAIR::JacobianVec(const vector<double>& v,
                        vector<double>& out,
                        vector<double>& out_add)
{

    double d0 = adjust_degrees_ ? 1 / sqrt_degrees_[0] : 1;
    double d1 = adjust_degrees_ ? 1 / sqrt_degrees_[1] : 1;
    double sum = d0 * d0 + d1 * d1;

    out.assign(2, 0.0);
    out_add.assign(1, 0.0);

    // first output
    if (!clip_u0_) {
        if ((branch_ == 0) || (branch_ == 1)) {
            out[0] += v[0];
        } else { // branch = 2: tied
            out[0] += v[0] * d0 * d0 / sum;
            out[0] += (flipped_ ? -1.0 : +1.0) * v[1] * d0 * d1 / sum;
        }
    }

    // second output: symmetrical
    if (!clip_u1_) {
        if ((branch_ == 0) || (branch_ == 1)) {
            out[1] += v[1];
        } else { // branch = 2: tied
            out[1] += (flipped_ ? -1.0 : +1.0) * v[0] * d0 * d1 / sum;
            out[1] += v[1] * (d1 * d1) / sum;
        }
    }

    // third column:
    if (!clip_u01_) {
        if (branch_ == 2) {
            out_add[0] += (v[0] * d0 + v[1] * d1) / sum;
        } else {
            if ((branch_ == 0) && !clip_u1_)
                out_add[0] += v[1] / d1;

            if (!clip_u0_ && (branch_ == 1 || (branch_ == 0 && flipped_)))
                out_add[0] += v[0] / d0;
        }
    }
}

} // namespace AD3
