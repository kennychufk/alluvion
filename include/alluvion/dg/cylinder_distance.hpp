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
        half_height_(height * static_cast<TF>(0.5)),
        com_y_(com_y),
        Distance<TF3, TF>(
            TF3{-radius, -height * static_cast<TF>(0.5) - com_y, -radius},
            TF3{radius, height * static_cast<TF>(0.5) - com_y, radius},
            sqrt(radius * radius +
                 (height * static_cast<TF>(0.5) + abs(com_y)) *
                     (height * static_cast<TF>(0.5) + abs(com_y)))) {}
  TF signedDistance(dg::Vector3r<TF> const& x) const override {
    TF d0 = sqrt(x(0) * x(0) + x(2) * x(2)) - radius_;
    TF d1 = abs(x(1) + com_y_) - half_height_;
    TF d0_clipped = max(d0, TF{0});
    TF d1_clipped = max(d1, TF{0});
    return min(max(d0, d1), TF{0}) +
           sqrt(d0_clipped * d0_clipped + d1_clipped * d1_clipped);
  }
  __device__ TF signed_distance(TF3 const& x) const {
    TF d0 = sqrt(x.x * x.x + x.z * x.z) - radius_;
    TF d1 = abs(x.y + com_y_) - half_height_;
    TF d0_clipped = max(d0, TF{0});
    TF d1_clipped = max(d1, TF{0});
    return min(max(d0, d1), TF{0}) +
           sqrt(d0_clipped * d0_clipped + d1_clipped * d1_clipped);
  }
  __device__ TF3 gradient(TF3 const& x, TF scale) const {
    constexpr TF kEps = 0.00390625;
    TF step = scale * kEps;
    return TF3{signed_distance(x + TF3{step, 0, 0}) -
                   signed_distance(x - TF3{step, 0, 0}),
               signed_distance(x + TF3{0, step, 0}) -
                   signed_distance(x - TF3{0, step, 0}),
               signed_distance(x + TF3{0, 0, step}) -
                   signed_distance(x - TF3{0, 0, step})};
  }

  TF radius_;
  TF half_height_;
  TF com_y_;
};
}  // namespace dg
}  // namespace alluvion

#endif /* ALLUVION_CYLINDER_DISTANCE_HPP */
