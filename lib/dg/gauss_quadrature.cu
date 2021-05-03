#include <algorithm>
#include <numeric>

#include "alluvion/dg/gauss_quadrature.hpp"

namespace alluvion {
namespace dg {

F GaussQuadrature::integrate(Integrand integrand, Domain const& domain,
                             unsigned int p) {
  if (p < 1) p = 1;

  // Number of Gauss points
  auto n = gaussian_n_1[p];

  auto c0 = (0.5 * domain.diagonal()).eval();
  auto c1 = (0.5 * (domain.min() + domain.max())).eval();

  auto res = 0.0;
  auto xi = Vector3r{};
  for (auto i = 0u; i < n; ++i) {
    auto wi = gaussian_weights_1[p][i];
    xi(0) = gaussian_abscissae_1[p][i];
    for (auto j = 0u; j < n; ++j) {
      auto wij = wi * gaussian_weights_1[p][j];
      xi(1) = gaussian_abscissae_1[p][j];
      for (auto k = 0u; k < n; ++k) {
        auto wijk = wij * gaussian_weights_1[p][k];
        xi(2) = gaussian_abscissae_1[p][k];
        res += wijk * integrand(c0.cwiseProduct(xi) + c1);
      }
    }
  }

  res *= c0.prod();
  return res;
}
}  // namespace dg
}  // namespace alluvion
