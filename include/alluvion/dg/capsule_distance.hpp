#ifndef ALLUVION_DG_CAPSULE_DISTANCE_HPP
#define ALLUVION_DG_CAPSULE_DISTANCE_HPP
#include "alluvion/dg/distance.hpp"

namespace alluvion {
namespace dg {
template <typename TF3, typename TF>
class CapsuleDistance : public Distance<TF3, TF> {
 public:
  CapsuleDistance(TF radius, TF height)
      : radius_(radius),
        half_height_(height * static_cast<TF>(0.5)),
        Distance<TF3, TF>(
            TF3{-radius, -height * static_cast<TF>(0.5) - radius, -radius},
            TF3{radius, height * static_cast<TF>(0.5) + radius, radius},
            height * static_cast<TF>(0.5) + radius) {}

  TF signedDistance(dg::Vector3r<TF> const& x) const override {
    TF3 x_mod{x(0), x(1) - max(-half_height_, min(x(1), half_height_)), x(2)};
    return length(x_mod) - radius_;
  }
  __device__ TF signed_distance(TF3 const& x) const {
    TF3 x_mod{x.x, x.y - max(-half_height_, min(x.y, half_height_)), x.z};
    return length(x_mod) - radius_;
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
};
}  // namespace dg
}  // namespace alluvion
#endif
