#ifndef ALLUVION_DG_DISTANCE_HPP
#define ALLUVION_DG_DISTANCE_HPP

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {
class Distance {
 public:
  Distance() : aabb_min_({0, 0, 0}), aabb_max_({0, 0, 0}){};
  Distance(F3 const& aabb_min, F3 const& aabb_max, F max_distance)
      : aabb_min_(aabb_min), aabb_max_(aabb_max), max_distance_(max_distance){};
  virtual ~Distance(){};
  virtual F signedDistance(dg::Vector3r const& x) const = 0;
  virtual F3 get_aabb_min() const { return aabb_min_; }
  virtual F3 get_aabb_max() const { return aabb_max_; }
  virtual F get_max_distance() const { return max_distance_; }
  F3 aabb_min_;
  F3 aabb_max_;
  F max_distance_;
};
}  // namespace dg
}  // namespace alluvion

#endif /* ALLUVION_DG_DISTANCE_HPP */
