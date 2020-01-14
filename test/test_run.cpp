#include <dynet/dict.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/globals.h>
#include <dynet/nodes.h>

#include <memory>

#include "ad3/FactorGraph.h"
#include "sparsemap.h"

#include <iostream>
#include <vector>

using std::vector;
namespace dy = dynet;

void
logic_factor()
{
    std::cout << "logic factor example\n";

    auto fg = std::make_unique<AD3::FactorGraph>();
    const size_t n_vars = 3;

    {
        vector<AD3::BinaryVariable*> vars;
        for (size_t i = 0; i < n_vars; ++i)
            vars.push_back(fg->CreateBinaryVariable());

        // fg->CreateFactorXOR(vars);
        fg->CreateFactorOROUT(vars);
    }

    dy::ComputationGraph cg;
    auto eta_u = dy::random_uniform(cg, { n_vars }, -1, 1);
    auto u = dy::sparsemap(eta_u, std::move(fg));

    auto z = dy::pick(u, (double)0);
    cg.forward(z);
    cg.backward(z, /*full=*/true);

    // std::cout << z.value() << std::endl;
    std::cout << "f(eta_u) =\n" << u.value() << std::endl;
    std::cout << "df[0] / deta_u =\n" << eta_u.gradient() << std::endl;
}

void
pair_factor_graph()
{
    std::cout << "pair graph example\n";

    auto fg = std::make_unique<AD3::FactorGraph>();

    {
        auto v_a = fg->CreateBinaryVariable();
        auto v_b = fg->CreateBinaryVariable();
        auto v_c = fg->CreateBinaryVariable();

        fg->CreateFactorPAIR({ v_a, v_b }, 0);
        fg->CreateFactorPAIR({ v_b, v_c }, 0);
    }

    dy::ComputationGraph cg;
    auto eta_u = dy::random_uniform(cg, { 3 }, -1, 1);
    auto eta_v = dy::random_uniform(cg, { 2 }, -1, 1);
    auto u = dy::sparsemap(eta_u, eta_v, std::move(fg));
    auto z = dy::pick(u, (double)0);

    cg.forward(z);
    cg.backward(z, /*full=*/true);

    std::cout << "f(eta_u) =\n" << u.value() << std::endl;
    std::cout << "df[0] / deta_u =\n" << eta_u.gradient() << std::endl;
    std::cout << "df[0] / deta_v =\n" << eta_v.gradient() << std::endl;
}

int
main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    dy::initialize(dyparams);

    logic_factor();

    pair_factor_graph();

    return 0;
}
