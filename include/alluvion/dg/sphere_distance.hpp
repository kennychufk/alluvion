#ifndef ALLUVION_DG_SPHERE_DISTANCE_HPP
#define ALLUVION_DG_SPHERE_DISTANCE_HPP

#include "alluvion/dg/distance.hpp"

namespace alluvion {
namespace dg {
class SphereDistance : public Distance {
 public:
  SphereDistance(F radius);
  F signedDistance(dg::Vector3r const& x) const override;
  F radius_;
};
}  // namespace dg
}  // namespace alluvion

#endif /* ALLUVION_DG_SPHERE_DISTANCE_HPP */
