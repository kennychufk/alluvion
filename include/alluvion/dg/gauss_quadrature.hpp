#ifndef ALLUVION_DG_GAUSS_QUADRATURE_HPP
#define ALLUVION_DG_GAUSS_QUADRATURE_HPP

#include <Eigen/Dense>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {
constexpr unsigned int gaussian_n_1[101] = {
    // {{{
    // p = 0
    0,
    // p = 1
    1,
    // p = 2
    2,
    // p = 3
    2,
    // p = 4
    3,
    // p = 5
    3,
    // p = 6
    4,
    // p = 7
    4,
    // p = 8
    5,
    // p = 9
    5,
    // p = 10
    6,
    // p = 11
    6,
    // p = 12
    7,
    // p = 13
    7,
    // p = 14
    8,
    // p = 15
    8,
    // p = 16
    9,
    // p = 17
    9,
    // p = 18
    10,
    // p = 19
    10,
    // p = 20
    11,
    // p = 21
    11,
    // p = 22
    12,
    // p = 23
    12,
    // p = 24
    13,
    // p = 25
    13,
    // p = 26
    14,
    // p = 27
    14,
    // p = 28
    15,
    // p = 29
    15,
    // p = 30
    16,
    // p = 31
    16,
    // p = 32
    17,
    // p = 33
    17,
    // p = 34
    18,
    // p = 35
    18,
    // p = 36
    19,
    // p = 37
    19,
    // p = 38
    20,
    // p = 39
    20,
    // p = 40
    21,
    // p = 41
    21,
    // p = 42
    22,
    // p = 43
    22,
    // p = 44
    23,
    // p = 45
    23,
    // p = 46
    24,
    // p = 47
    24,
    // p = 48
    25,
    // p = 49
    25,
    // p = 50
    26,
    // p = 51
    26,
    // p = 52
    27,
    // p = 53
    27,
    // p = 54
    28,
    // p = 55
    28,
    // p = 56
    29,
    // p = 57
    29,
    // p = 58
    30,
    // p = 59
    30,
    // p = 60
    31,
    // p = 61
    31,
    // p = 62
    32,
    // p = 63
    32,
    // p = 64
    33,
    // p = 65
    33,
    // p = 66
    34,
    // p = 67
    34,
    // p = 68
    35,
    // p = 69
    35,
    // p = 70
    36,
    // p = 71
    36,
    // p = 72
    37,
    // p = 73
    37,
    // p = 74
    38,
    // p = 75
    38,
    // p = 76
    39,
    // p = 77
    39,
    // p = 78
    40,
    // p = 79
    40,
    // p = 80
    41,
    // p = 81
    41,
    // p = 82
    42,
    // p = 83
    42,
    // p = 84
    43,
    // p = 85
    43,
    // p = 86
    44,
    // p = 87
    44,
    // p = 88
    45,
    // p = 89
    45,
    // p = 90
    46,
    // p = 91
    46,
    // p = 92
    47,
    // p = 93
    47,
    // p = 94
    48,
    // p = 95
    48,
    // p = 96
    49,
    // p = 97
    49,
    // p = 98
    50,
    // p = 99
    50,
    // p = 100
    51
    // }}}
};

template <typename TF>
struct GaussConst {
  // TODO: optimize precision
  constexpr GaussConst() : abscissae(), weights() {
    abscissae[30][0] = -0.989400934991649938510249739920;
    abscissae[30][1] = -0.944575023073232600268056557979;
    abscissae[30][2] = -0.865631202387831755196145877562;
    abscissae[30][3] = -0.755404408355002998654015300417;
    abscissae[30][4] = -0.617876244402643770570193737512;
    abscissae[30][5] = -0.458016777657227369680015272024;
    abscissae[30][6] = -0.281603550779258915426339626720;
    abscissae[30][7] = -0.095012509837637426635126303154;
    abscissae[30][8] = 0.095012509837637426635126303154;
    abscissae[30][9] = 0.281603550779258915426339626720;
    abscissae[30][10] = 0.458016777657227369680015272024;
    abscissae[30][11] = 0.617876244402643770570193737512;
    abscissae[30][12] = 0.755404408355002998654015300417;
    abscissae[30][13] = 0.865631202387831755196145877562;
    abscissae[30][14] = 0.944575023073232600268056557979;
    abscissae[30][15] = 0.989400934991649938510249739920;

    weights[30][0] = 0.027152459411758110563450685504;
    weights[30][1] = 0.062253523938649010793788818319;
    weights[30][2] = 0.095158511682492036287683845330;
    weights[30][3] = 0.124628971255533488315947465708;
    weights[30][4] = 0.149595988816575764523975067277;
    weights[30][5] = 0.169156519395001675443168664970;
    weights[30][6] = 0.182603415044922529064663763165;
    weights[30][7] = 0.189450610455067447457366824892;
    weights[30][8] = 0.189450610455067447457366824892;
    weights[30][9] = 0.182603415044922529064663763165;
    weights[30][10] = 0.169156519395001675443168664970;
    weights[30][11] = 0.149595988816575764523975067277;
    weights[30][12] = 0.124628971255533488315947465708;
    weights[30][13] = 0.095158511682492036287683845330;
    weights[30][14] = 0.062253523938649010793788818319;
    weights[30][15] = 0.027152459411758110563450685504;
  }
  TF abscissae[101][51];
  TF weights[101][51];
};

template <typename TF>
class GaussQuadrature {
 public:
  using Integrand = std::function<TF(Vector3r<TF> const&)>;
  using Domain = AlignedBox3r<TF>;

  static TF integrate(Integrand integrand, Domain const& domain,
                      unsigned int p) {
    if (p < 1) p = 1;

    // Number of Gauss points
    auto n = gaussian_n_1[p];

    auto c0 = (0.5 * domain.diagonal()).eval();
    auto c1 = (0.5 * (domain.min() + domain.max())).eval();

    auto res = 0.0;
    auto xi = Vector3r<TF>{};
    constexpr auto kGaussConst = GaussConst<TF>();
    for (auto i = 0u; i < n; ++i) {
      auto wi = kGaussConst.weights[p][i];
      xi(0) = kGaussConst.abscissae[p][i];
      for (auto j = 0u; j < n; ++j) {
        auto wij = wi * kGaussConst.weights[p][j];
        xi(1) = kGaussConst.abscissae[p][j];
        for (auto k = 0u; k < n; ++k) {
          auto wijk = wij * kGaussConst.weights[p][k];
          xi(2) = kGaussConst.abscissae[p][k];
          res += wijk * integrand(c0.cwiseProduct(xi) + c1);
        }
      }
    }

    res *= c0.prod();
    return res;
  }
};
}  // namespace dg
}  // namespace alluvion

#endif
