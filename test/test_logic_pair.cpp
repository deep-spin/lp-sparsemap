#include <dynet/dict.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/globals.h>
#include <dynet/grad-check.h>
#include <dynet/nodes.h>

#include <memory>

#include "ad3/FactorGraph.h"
#include "sparsemap.h"

#include <iostream>
#include <vector>

using std::vector;
namespace dy = dynet;

std::unique_ptr<AD3::FactorGraph>
make_logic_pair_fg(int n_rows, int n_cols)
{
    auto fg = std::make_unique<AD3::FactorGraph>();

    std::vector<AD3::BinaryVariable*> vars;
    std::vector<AD3::BinaryVariable*> vars_col(n_rows);

    for (int j = 0; j < n_cols; ++j) {
        for (int i = 0; i < n_rows; ++i) {
            auto var = fg->CreateBinaryVariable();
            vars.push_back(var);
            vars_col.at(i) = var;
        }
        fg->CreateFactorXOR(vars_col);
    }

    {
        auto a_i = 0, a_j = 0;
        auto b_j = 1;
        for (int b_i = 0; b_i < n_rows; ++b_i) {
            auto v_ij = vars.at(n_rows * a_j + a_i);
            auto v_ipjp = vars.at(n_rows * b_j + b_i);
            fg->CreateFactorPAIR({ v_ij, v_ipjp }, .0f);
        }
    }
    /*
    for (int j = 0; j < n_cols; ++j) {
        for (int i = 0; i < n_rows; ++i) {
            auto ip = (i + 1) % n_rows;
            auto jp = (j + 1) % n_cols;

            auto v_ij = vars.at(n_rows * j + i);
            auto v_ipjp = vars.at(n_rows * jp + ip);
            fg->CreateFactorPAIR({ v_ij, v_ipjp }, .0f);
        }
    }
    */

    return fg;
}

void
test_logic_pair(unsigned int n_rows, unsigned int n_cols)
{
    std::cout << "Logic & Pair" << std::endl;

    dy::SparseMAPOpts opts;
    opts.max_iter = 100000;
    opts.residual_thr = 1e-12;
    opts.max_iter_backward = 100000;
    opts.atol_thr_backward = 1e-12;
    opts.max_active_set_iter = 100;

    dy::ParameterCollection params;

    unsigned n_u = n_rows * n_cols;
    unsigned n_add = 0;
    {
        auto fg = make_logic_pair_fg(n_rows, n_cols);
        fg->Print(cout);
        n_add = fg->GetNumFactors();
    }
    n_add -= n_cols;
    auto eta_u = params.add_parameters({ n_u }, 1, "etau");
    auto eta_v = params.add_parameters({ n_add }, 1, "etav");

    {
        dy::ComputationGraph cg;
        auto e_eta_u = dy::parameter(cg, eta_u);
        auto e_eta_v = dy::parameter(cg, eta_v);
        auto u = dy::sparsemap(
          e_eta_u,
          e_eta_v,
          std::move(make_logic_pair_fg(n_rows, n_cols)),
          opts);

        u = dy::reshape(u, { n_rows, n_cols });
        std::cout << u.value() << std::endl;
    }

    unsigned offset = 0;
    for (int j = 0; j < n_cols; ++j) {
        for (int i = 0; i < n_rows; ++i) {
            dy::ComputationGraph cg;
            auto e_eta_u = dy::parameter(cg, eta_u);
            auto e_eta_v = dy::parameter(cg, eta_v);

            auto u = dy::sparsemap(
              e_eta_u,
              e_eta_v,
              std::move(make_logic_pair_fg(n_rows, n_cols)),
              opts);

            auto z = dy::pick(u, offset);
            cg.backward(z);
            check_grad(params, z, /*verbosity=*/1);

            offset += 1;
        }
    }
}

int
main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    dy::initialize(dyparams);

    test_logic_pair(3, 2);

    return 0;
}
