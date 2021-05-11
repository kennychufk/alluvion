#ifndef ALLUVION_DG_INFINITE_CYLINDER_DISTANCE_HPP
#define ALLUVION_DG_INFINITE_CYLINDER_DISTANCE_HPP

#include "alluvion/dg/distance.hpp"

namespace alluvion {
namespace dg {
class InfiniteCylinderDistance : public Distance {
 public:
  InfiniteCylinderDistance(F radius);
  F signedDistance(dg::Vector3r const& x) const override;
  F radius_;
};
}  // namespace dg
}  // namespace alluvion

#endif /* ALLUVION_DG_INFINITE_CYLINDER_DISTANCE_HPP */
