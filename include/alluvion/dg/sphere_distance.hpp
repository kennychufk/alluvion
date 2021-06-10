#ifndef ALLUVION_DG_SPHERE_DISTANCE_HPP
#define ALLUVION_DG_SPHERE_DISTANCE_HPP

#include "alluvion/dg/distance.hpp"

namespace alluvion {
namespace dg {
template <typename TF3, typename TF>
class SphereDistance : public Distance<TF3, TF> {
 public:
  SphereDistance(TF radius)
      : radius_(radius),
        Distance<TF3, TF>(TF3{-radius, -radius, -radius},
                          TF3{radius, radius, radius}, radius){};
  TF signedDistance(dg::Vector3r<TF> const& x) const override {
    return x.norm() - radius_;
  }
  TF radius_;
};
}  // namespace dg
}  // namespace alluvion

#endif /* ALLUVION_DG_SPHERE_DISTANCE_HPP */
