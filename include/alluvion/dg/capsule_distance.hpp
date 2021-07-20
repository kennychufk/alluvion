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
  TF radius_;
  TF half_height_;
};
}  // namespace dg
}  // namespace alluvion
#endif
