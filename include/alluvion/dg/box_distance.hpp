#ifndef ALLUVION_BOX_DISTANCE_HPP
#define ALLUVION_BOX_DISTANCE_HPP

#include "alluvion/dg/distance.hpp"

namespace alluvion {
namespace dg {
template <typename TF3, typename TF>
class BoxDistance : public Distance<TF3, TF> {
 public:
  BoxDistance(TF3 widths)
      : half_widths(widths * TF{0.5}),
        Distance<TF3, TF>(widths * TF{-0.5}, widths * TF{0.5}, length(widths)) {
  }
  TF signedDistance(dg::Vector3r<TF> const& x) const override {
    TF3 diff = TF3{abs(x(0)), abs(x(1)), abs(x(2))} - half_widths;
    TF3 clipped_diff =
        TF3{max(diff.x, TF{0}), max(diff.y, TF{0}), max(diff.z, TF{0})};
    return length(clipped_diff) + min(max(diff.x, max(diff.y, diff.z)), TF{0});
  }
  TF3 half_widths;
};
}  // namespace dg
}  // namespace alluvion
#endif /* ALLUVION_BOX_DISTANCE_HPP */