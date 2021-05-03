#ifndef ALLUVION_DG_SPH_KERNELS_HPP
#define ALLUVION_DG_SPH_KERNELS_HPP

#include <algorithm>
#include <cmath>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {
/** \brief Cubic spline kernel.
 */
class CubicKernel {
 protected:
  static F m_radius;
  static F m_k;
  static F m_l;
  static F m_W_zero;

 public:
  static F getRadius() { return m_radius; }
  static void setRadius(F val) {
    m_radius = val;
    const F pi = static_cast<F>(M_PI);

    const F h3 = m_radius * m_radius * m_radius;
    m_k = static_cast<F>(8.0) / (pi * h3);
    m_l = static_cast<F>(48.0) / (pi * h3);
    m_W_zero = W(Vector3r::Zero());
  }

 public:
  static F W(const F r) {
    F res = 0.0;
    const F q = r / m_radius;
    if (q <= 1.0) {
      if (q <= 0.5) {
        const F q2 = q * q;
        const F q3 = q2 * q;
        res = m_k * (static_cast<F>(6.0) * q3 - static_cast<F>(6.0) * q2 +
                     static_cast<F>(1.0));
      } else {
        res = m_k * (static_cast<F>(2.0) * pow(static_cast<F>(1.0) - q, 3));
      }
    }
    return res;
  }

  static F W(const Vector3r &r) { return W(r.norm()); }

  static Vector3r gradW(const Vector3r &r) {
    Vector3r res;
    const F rl = r.norm();
    const F q = rl / m_radius;
    if ((rl > 1.0e-5) && (q <= 1.0)) {
      const Vector3r gradq = r * (static_cast<F>(1.0) / (rl * m_radius));
      if (q <= 0.5) {
        res = m_l * q * ((F)3.0 * q - static_cast<F>(2.0)) * gradq;
      } else {
        const F factor = static_cast<F>(1.0) - q;
        res = m_l * (-factor * factor) * gradq;
      }
    } else
      res.setZero();

    return res;
  }

  static F W_zero() { return m_W_zero; }
};

}  // namespace dg
}  // namespace alluvion
#endif
