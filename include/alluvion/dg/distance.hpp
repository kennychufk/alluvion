#ifndef ALLUVION_DG_DISTANCE_HPP
#define ALLUVION_DG_DISTANCE_HPP

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {
class Distance {
 public:
  Distance(){};
  virtual ~Distance(){};
  virtual F signedDistance(dg::Vector3r const& x) const = 0;
};
}  // namespace dg
}  // namespace alluvion

#endif /* ALLUVION_DG_DISTANCE_HPP */
