#ifndef ALLUVION_DG_INFINITE_TUBE_DISTANCE_HPP
#define ALLUVION_DG_INFINITE_TUBE_DISTANCE_HPP

#include "alluvion/dg/distance.hpp"
namespace alluvion {
namespace dg {
template <typename TF3, typename TF>
class InfiniteTubeDistance : public Distance<TF3, TF> {
 public:
  InfiniteTubeDistance(TF inner_radius, TF outer_radius,
                       TF aabb_half_length = 0)
      : inner_radius_(inner_radius),
        outer_radius_(outer_radius),
        Distance<TF3, TF>(
            TF3{-outer_radius,
                aabb_half_length == 0 ? -outer_radius : -aabb_half_length,
                -outer_radius},
            TF3{outer_radius,
                aabb_half_length == 0 ? outer_radius : aabb_half_length,
                outer_radius},
            max(aabb_half_length, outer_radius)) {}

  TF signedDistance(dg::Vector3r<TF> const& x) const override {
    TF radial_distance = sqrt(x(0) * x(0) + x(2) * x(2));
    return max(inner_radius_ - radial_distance,
               radial_distance - outer_radius_);
  }
  __device__ TF signed_distance(TF3 const& x) const {
    TF radial_distance = sqrt(x.x * x.x + x.z * x.z);
    return max(inner_radius_ - radial_distance,
               radial_distance - outer_radius_);
  }
  __device__ TF3 gradient(TF3 const& x, TF scale) const {
    return ((x.x * x.x + x.z * x.z) * 4 > (inner_radius_ + outer_radius_) *
                                              (inner_radius_ + outer_radius_)
                ? 1
                : -1) *
           TF3{x.x, 0, x.z};
  }
  TF inner_radius_;
  TF outer_radius_;
};
}  // namespace dg
}  // namespace alluvion

#endif
