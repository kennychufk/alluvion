#ifndef ALLUVION_DG_SPH_KERNELS_HPP
#define ALLUVION_DG_SPH_KERNELS_HPP

#include <algorithm>
#include <cmath>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {
/** \brief Cubic spline kernel.
 */
template <typename TF>
class CubicKernel {
 protected:
  TF m_radius;
  TF m_k;
  TF m_l;
  TF m_W_zero;

 public:
  TF getRadius() { return m_radius; }
  void setRadius(TF val) {
    m_radius = val;
    const TF pi = static_cast<TF>(M_PI);

    const TF h3 = m_radius * m_radius * m_radius;
    m_k = static_cast<TF>(8.0) / (pi * h3);
    m_l = static_cast<TF>(48.0) / (pi * h3);
    m_W_zero = W(Vector3r<TF>::Zero());
  }

 public:
  TF W(const TF r) {
    TF res = 0.0;
    const TF q = r / m_radius;
    if (q <= 1.0) {
      if (q <= 0.5) {
        const TF q2 = q * q;
        const TF q3 = q2 * q;
        res = m_k * (static_cast<TF>(6.0) * q3 - static_cast<TF>(6.0) * q2 +
                     static_cast<TF>(1.0));
      } else {
        res = m_k * (static_cast<TF>(2.0) * pow(static_cast<TF>(1.0) - q, 3));
      }
    }
    return res;
  }

  TF W(const Vector3r<TF> &r) { return W(r.norm()); }

  Vector3r<TF> gradW(const Vector3r<TF> &r) {
    Vector3r<TF> res;
    const TF rl = r.norm();
    const TF q = rl / m_radius;
    if ((rl > 1.0e-5) && (q <= 1.0)) {
      const Vector3r<TF> gradq = r * (static_cast<TF>(1.0) / (rl * m_radius));
      if (q <= 0.5) {
        res = m_l * q * ((TF)3.0 * q - static_cast<TF>(2.0)) * gradq;
      } else {
        const TF factor = static_cast<TF>(1.0) - q;
        res = m_l * (-factor * factor) * gradq;
      }
    } else
      res.setZero();

    return res;
  }

  TF W_zero() { return m_W_zero; }
};

}  // namespace dg
}  // namespace alluvion
#endif
