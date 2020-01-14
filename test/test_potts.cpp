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
make_potts_fg(int n_rows, int n_cols, int n_states)
{
    auto fg = std::make_unique<AD3::FactorGraph>();

    vector<AD3::MultiVariable*> vars;

    for (int i = 0; i < n_rows; ++i)
        for (int j = 0; j < n_cols; ++j)
            vars.push_back(fg->CreateMultiVariable(n_states));

    vector<double> zeros(n_states * n_states);

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 1; j < n_cols; ++j) {

            int a = i * n_cols + j - 1;
            int b = i * n_cols + j;

            fg->CreateFactorDense({ vars[a], vars[b] }, zeros);
        }
    }

    for (int j = 0; j < n_cols; ++j) {
        for (int i = 1; i < n_rows; ++i) {

            int a = (i - 1) * n_cols + j;
            int b = i * n_cols + j;

            fg->CreateFactorDense({ vars[a], vars[b] }, zeros);
        }
    }

    return fg;
}

void
test_potts(int n_rows, int n_cols, int n_states)
{
    std::cout << "Potts factor\n"
              << n_rows << " rows, " << n_cols << " columns, " << n_states
              << " states." << std::endl;

    dy::SparseMAPOpts opts;
    opts.max_iter = 100000;
    opts.residual_thr = 1e-12;
    opts.max_iter_backward = 1000;
    opts.atol_thr_backward = 1e-8;
    opts.max_active_set_iter = 100;
    opts.log_stream = std::make_shared<std::ofstream>("log.txt");

    dy::ParameterCollection params;

    unsigned n_u = n_rows * n_cols * n_states;
    unsigned n_add =
      (2 * n_rows * n_cols - n_rows - n_cols) * n_states * n_states;
    auto eta_u = params.add_parameters({ n_u }, 1, "etau");
    auto eta_v = params.add_parameters({ n_add }, 1, "etav");

    {
        dy::ComputationGraph cg;
        auto e_eta_u = dy::parameter(cg, eta_u);
        auto e_eta_v = dy::parameter(cg, eta_v);
        auto u = dy::sparsemap(
          e_eta_u,
          e_eta_v,
          std::move(make_potts_fg(n_rows, n_cols, n_states)),
          opts);

        std::cout << u.value() << std::endl;
    }

    opts.log_stream.reset();

    unsigned offset = 0;
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            for (int k = 0; k < n_states; ++k) {
                dy::ComputationGraph cg;
                auto e_eta_u = dy::parameter(cg, eta_u);
                auto e_eta_v = dy::parameter(cg, eta_v);

                auto u = dy::sparsemap(
                  e_eta_u,
                  e_eta_v,
                  std::move(make_potts_fg(n_rows, n_cols, n_states)),
                  opts);

                auto z = dy::pick(u, offset);
                cg.backward(z);
                check_grad(params, z, /*verbosity=*/1);

                offset += 1;
            }
        }
    }
}

int
main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    dy::initialize(dyparams);

    test_potts(3, 2, 3);

    return 0;
}
