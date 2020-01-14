#include <dynet/dynet.h>
#include <dynet/grad-check.h>
#include <memory>

#include "sparsemap.h"
#include "ad3/FactorGraph.h"

#include <iostream>

namespace dy = dynet;

std::unique_ptr<AD3::FactorGraph>
make_fg(int verbose=0)
{
    auto fg = std::make_unique<AD3::FactorGraph>();
    auto a = fg->CreateBinaryVariable();
    auto b = fg->CreateBinaryVariable();
    auto c = fg->CreateBinaryVariable();
    auto f = fg->CreateFactorXOR({ a, b });
    auto g = fg->CreateFactorXOR({ b, c });

    fg->SetVerbosity(verbose);

    return fg;

}

void test_logic()
{
    dy::SparseMAPOpts opts;
    opts.max_iter = 100000;
    opts.residual_thr = 1e-12;
    opts.max_iter_backward = 100000;
    opts.atol_thr_backward = 1e-12;
    opts.max_active_set_iter = 100;

    dy::ParameterCollection params;

    unsigned int n_vars = 3;

    auto eta_u = params.add_parameters({ n_vars }, 1u, "etau");
    {
        dy::ComputationGraph cg;
        auto e_eta_u = dy::parameter(cg, eta_u);

        auto u = dy::sparsemap(
          e_eta_u,
          std::move(make_fg(10)),
          opts
        );
        std::cout << u.value() << std::endl;
    }

    for (int k = 0; k < n_vars; ++k) {
    //for (int k = 1; k < 2; ++k) {
        std::cout << "wrt output " <<  k << std::endl;
        dy::ComputationGraph cg;
        auto e_eta_u = dy::parameter(cg, eta_u);

        auto u = dy::sparsemap(
          e_eta_u,
          std::move(make_fg()),
          opts
        );

        auto z = dy::pick(u, (unsigned) k);
        check_grad(params, z, /*verbosity=*/2);
    }
}

int
main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    dy::initialize(dyparams);

    test_logic();

    return 0;
}
