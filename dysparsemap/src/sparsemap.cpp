#include <dynet/nodes-impl-macros.h>
#include <dynet/tensor-eigen.h>

#include "ad3/FactorGraph.h"
#include "sparsemap.h"

#include <iostream>

namespace dynet {

Expression
sparsemap(const Expression& eta_u,
          std::unique_ptr<AD3::FactorGraph> fg,
          const SparseMAPOpts& opts)
{
    auto cg = eta_u.pg;
    return Expression(
      cg, cg->add_function<SparseMAP>({ eta_u.i }, std::move(fg), opts));
}

Expression
sparsemap(const dynet::Expression& eta_u,
          const dynet::Expression& eta_v,
          std::unique_ptr<AD3::FactorGraph> fg,
          const SparseMAPOpts& opts)
{
    auto cg = eta_u.pg;
    return Expression(
      cg,
      cg->add_function<SparseMAP>({ eta_u.i, eta_v.i }, std::move(fg), opts));
}

Expression
sparsemap(const Expression& eta_u, std::unique_ptr<AD3::FactorGraph> fg)
{
    SparseMAPOpts opts;
    return sparsemap(eta_u, std::move(fg), opts);
}

Expression
sparsemap(const Expression& eta_u,
          const Expression& eta_v,
          std::unique_ptr<AD3::FactorGraph> fg)
{
    SparseMAPOpts opts;
    return sparsemap(eta_u, eta_v, std::move(fg), opts);
}

SparseMAP::SparseMAP(const std::initializer_list<VariableIndex>& a,
                     std::unique_ptr<AD3::FactorGraph> moved_fg,
                     const SparseMAPOpts& opts)
  : Node(a)
  , opts(opts)
  , fg(std::move(moved_fg))
  , n_vars(fg->GetNumVariables())
  , n_factors(fg->GetNumFactors())
{
    // count the number of additionals, for convenience and error checking
    for (size_t f = 0; f < n_factors; ++f) {
        n_add += fg->GetFactor(f)->GetNumAdditionals();
    }

    // std::cout << "constructing " << n_vars << " " << n_factors
    //<< " " << n_add << std::endl;
    // std::abort();
}

std::string
SparseMAP::as_string(const std::vector<std::string>& arg_names) const
{
    std::ostringstream s;
    s << "sparsemap(";
    for (auto&& arg_name : arg_names)
        s << arg_name << ", ";
    s << ")";
    return s.str();
}

// sparsemap : (eta_u, eta_v) -> u, with dim(u) = dim(eta_u)
Dim
SparseMAP::dim_forward(const std::vector<Dim>& xs) const
{
    return xs[0];
}

template<class MyDevice>
void
SparseMAP::forward_dev_impl(const MyDevice&,
                            const std::vector<const Tensor*>& xs,
                            Tensor& fx) const
{
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("SparseMAP::forward");
#else

    // set variable log potentials
    auto eta_u = vec(*xs[0]);
    for (size_t i = 0; i < n_vars; ++i)
        fg->GetBinaryVariable(i)->SetLogPotential(eta_u(i));

    // set additional log potentials
    if (n_add > 0) {
        auto ptr = xs[1]->v;
        for (size_t k = 0; k < n_factors; ++k) {
            auto f = fg->GetFactor(k);
            size_t f_n_add = f->GetNumAdditionals();
            if (f_n_add > 0) {
                f->SetAdditionalLogPotentials(
                  vector<double>(ptr, ptr + f_n_add));
                ptr += f_n_add;
            }
        }
    }

    auto out = vec(fx);
    out.setZero();

    vector<double> u(n_vars), v(n_add);
    double val;

    fg->SetMaxIterationsAD3(opts.max_iter);
    fg->SetEtaAD3(opts.eta);
    fg->AdaptEtaAD3(opts.adapt_eta);
    fg->SetResidualThresholdAD3(opts.residual_thr);

    for (size_t k = 0; k < n_factors; ++k) {
        auto f = fg->GetFactor(k);
        if (f->IsGeneric()) {
            auto gf = static_cast<AD3::GenericFactor*>(f);
            gf->SetQPMaxIter(opts.max_active_set_iter);
        }
    }

    fg->SolveQP(&u, &v, &val);
    if (opts.log_stream)
        fg->PrintStructures(*opts.log_stream);

    for (size_t i = 0; i < n_vars; ++i)
        out(i) = u[i];

#endif
}

template<class MyDevice>
void
SparseMAP::backward_dev_impl(const MyDevice&,
                             const std::vector<const Tensor*>&,
                             const Tensor&,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const
{
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("SparseMAP::backward");
#else

    auto dEdxi_v = vec(dEdxi);

    vector<double> in(dEdf.v, dEdf.v + n_vars);
    vector<double> out(n_vars);

    auto dv = fg->JacobianVec(in.data(),
                              out.data(),
                              opts.max_iter_backward,
                              opts.atol_thr_backward);

    if (i == 0) {
        for (size_t ii = 0; ii < n_vars; ++ii)
            dEdxi_v(ii) += out[ii];
    } else if (i == 1) {
        size_t ii = 0;
        for (size_t k = 0; k < n_factors; ++k) {
            for (size_t j = 0; j < dv[k].size(); ++j) {
                dEdxi_v(ii) += dv[k][j];
                ii += 1;
            }
        }
    }
#endif
}

DYNET_NODE_INST_DEV_IMPL(SparseMAP)

}
