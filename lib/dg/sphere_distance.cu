#include "alluvion/dg/sphere_distance.hpp"

namespace alluvion {
namespace dg {
SphereDistance::SphereDistance(F radius) : radius_(radius) {}

F SphereDistance::signedDistance(dg::Vector3r const& x) const {
  return x.norm() - radius_;
}

}  // namespace dg
}  // namespace alluvion
