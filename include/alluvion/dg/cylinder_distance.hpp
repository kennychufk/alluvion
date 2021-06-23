#ifndef ALLUVION_CYLINDER_DISTANCE_HPP
#define ALLUVION_CYLINDER_DISTANCE_HPP

#include "alluvion/dg/distance.hpp"

namespace alluvion {
namespace dg {
template <typename TF3, typename TF>
class CylinderDistance : public Distance<TF3, TF> {
 public:
  CylinderDistance(TF radius, TF height, TF com_y = 0)
      : radius_(radius),
        height_(height),
        com_y_(com_y),
        Distance<TF3, TF>(
            TF3{-radius, -height / 2 - com_y, -radius},
            TF3{radius, height / 2 - com_y, radius},
            sqrt(radius * radius +
                 (height / 2 + abs(com_y)) * (height / 2 + abs(com_y)))) {}
  TF signedDistance(dg::Vector3r<TF> const& x) const override {
    TF d0 = sqrt(x(0) * x(0) + x(2) * x(2)) - radius_;
    TF d1 = abs(x(1) + com_y_) - height_ * static_cast<TF>(0.5);
    TF d0_clipped = max(d0, TF{0});
    TF d1_clipped = max(d1, TF{0});
    return min(max(d0, d1), TF{0}) +
           sqrt(d0_clipped * d0_clipped + d1_clipped * d1_clipped);
  }

  TF radius_;
  TF height_;
  TF com_y_;
};
}  // namespace dg
}  // namespace alluvion

#endif /* ALLUVION_CYLINDER_DISTANCE_HPP */
