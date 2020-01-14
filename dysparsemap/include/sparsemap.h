#pragma once

#include <fstream>

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/nodes-def-macros.h>
#include <dynet/nodes.h>

#include "ad3/FactorGraph.h"

namespace dynet {

struct SparseMAPOpts
{
    /* forward */
    int max_iter = 1000;
    double eta = 0.1;
    bool adapt_eta = true;
    double residual_thr = 1e-6;

    /* backward */
    int max_iter_backward = 100;
    double atol_thr_backward = 1e-8;

    /* QP active set */
    int max_active_set_iter = 10;

    std::shared_ptr<std::ofstream> log_stream;
};

dynet::Expression
sparsemap(const dynet::Expression& eta_u,
          std::unique_ptr<AD3::FactorGraph> fg);

dynet::Expression
sparsemap(const dynet::Expression& eta_u,
          std::unique_ptr<AD3::FactorGraph> fg,
          const SparseMAPOpts& opts);

dynet::Expression
sparsemap(const dynet::Expression& eta_u,
          const dynet::Expression& eta_v,
          std::unique_ptr<AD3::FactorGraph> fg);

dynet::Expression
sparsemap(const dynet::Expression& eta_u,
          const dynet::Expression& eta_v,
          std::unique_ptr<AD3::FactorGraph> fg,
          const SparseMAPOpts& opts);

struct SparseMAP : public dynet::Node
{
    const SparseMAPOpts opts;
    std::unique_ptr<AD3::FactorGraph> fg;
    size_t n_vars, n_factors;
    size_t n_add = 0;

    explicit SparseMAP(const std::initializer_list<dynet::VariableIndex>&,
                       std::unique_ptr<AD3::FactorGraph>,
                       const SparseMAPOpts& opts);

    DYNET_NODE_DEFINE_DEV_IMPL()
};

}
