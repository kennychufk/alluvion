#ifndef ALLUVION_DG_DISTANCE_HPP
#define ALLUVION_DG_DISTANCE_HPP

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {
template <typename TF3, typename TF>
class Distance {
 public:
  Distance() : aabb_min_({0, 0, 0}), aabb_max_({0, 0, 0}) {}
  Distance(TF3 const& aabb_min, TF3 const& aabb_max, TF max_distance)
      : aabb_min_(aabb_min), aabb_max_(aabb_max), max_distance_(max_distance) {}
  virtual ~Distance(){};
  virtual TF signedDistance(dg::Vector3r<TF> const& x) const = 0;
  virtual TF3 get_aabb_min() const { return aabb_min_; }
  virtual TF3 get_aabb_max() const { return aabb_max_; }
  virtual TF get_max_distance() const { return max_distance_; }
  TF3 aabb_min_;
  TF3 aabb_max_;
  TF max_distance_;
};
}  // namespace dg
}  // namespace alluvion

#endif /* ALLUVION_DG_DISTANCE_HPP */
