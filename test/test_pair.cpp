#include <dynet/dynet.h>
#include <dynet/grad-check.h>
#include <memory>

#include "sparsemap.h"
#include "ad3/FactorGraph.h"

#include <iostream>

namespace dy = dynet;

void test_pair()
{
    dy::SparseMAPOpts opts;
    opts.max_iter = 100000;
    opts.residual_thr = 1e-12;
    opts.max_iter_backward = 100000;
    opts.atol_thr_backward = 1e-12;
    opts.max_active_set_iter = 100;

    dy::ParameterCollection params;

    auto eta_u = params.add_parameters({ 3 }, 1, "etau");
    auto eta_v = params.add_parameters({ 2 }, 1, "etav");

    for (int k = 0; k < 3; ++k) {
        std::cout << "wrt output " <<  k << std::endl;
        dy::ComputationGraph cg;
        auto e_eta_u = dy::parameter(cg, eta_u);
        auto e_eta_v = dy::parameter(cg, eta_v);

        auto fg = std::make_unique<AD3::FactorGraph>();
        auto a = fg->CreateBinaryVariable();
        auto b = fg->CreateBinaryVariable();
        auto c = fg->CreateBinaryVariable();
        auto f = fg->CreateFactorPAIR({ a, b }, 0.0f);
        auto g = fg->CreateFactorPAIR({ b, c }, 0.0f);

        auto u = dy::sparsemap(
          e_eta_u,
          e_eta_v,
          std::move(fg),
          opts
        );

        auto z = dy::pick(u, (unsigned) k);
        check_grad(params, z, /*verbosity=*/1);
    }
}

int
main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    dy::initialize(dyparams);

    test_pair();

    return 0;
}
