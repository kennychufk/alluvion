#include "alluvion/dg/infinite_cylinder_distance.hpp"

namespace alluvion {
namespace dg {
InfiniteCylinderDistance::InfiniteCylinderDistance(F radius)
    : radius_(radius),
      // AABB should be finite
      Distance(F3{-radius, -radius, -radius}, F3{radius, radius, radius},
               radius) {}

F InfiniteCylinderDistance::signedDistance(dg::Vector3r const& x) const {
  return sqrt(x(1) * x(1) + x(2) * x(2)) - radius_;
}

}  // namespace dg
}  // namespace alluvion
