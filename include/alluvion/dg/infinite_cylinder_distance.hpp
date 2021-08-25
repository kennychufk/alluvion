#ifndef ALLUVION_DG_INFINITE_CYLINDER_DISTANCE_HPP
#define ALLUVION_DG_INFINITE_CYLINDER_DISTANCE_HPP

#include "alluvion/dg/distance.hpp"

namespace alluvion {
namespace dg {
template <typename TF3, typename TF>
class InfiniteCylinderDistance : public Distance<TF3, TF> {
 public:
  InfiniteCylinderDistance(TF radius)
      : radius_(radius),
        // AABB should be finite
        Distance<TF3, TF>(TF3{-radius, -radius, -radius},
                          TF3{radius, radius, radius}, radius) {}

  TF signedDistance(dg::Vector3r<TF> const& x) const override {
    return sqrt(x(0) * x(0) + x(2) * x(2)) - radius_;
  }
  __device__ TF signed_distance(TF3 const& x) const {
    return sqrt(x.x * x.x + x.z * x.z) - radius_;
  }
  __device__ TF3 gradient(TF3 const& x, TF scale) const {
    return TF3{x.x, 0, x.z};
  }
  TF radius_;
};
}  // namespace dg
}  // namespace alluvion

#endif /* ALLUVION_DG_INFINITE_CYLINDER_DISTANCE_HPP */
