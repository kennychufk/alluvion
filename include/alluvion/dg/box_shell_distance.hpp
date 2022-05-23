#ifndef ALLUVION_BOX_SHELL_DISTANCE_HPP
#define ALLUVION_BOX_SHELL_DISTANCE_HPP

#include "alluvion/dg/distance.hpp"

namespace alluvion {
namespace dg {
template <typename TF3, typename TF>
class BoxShellDistance : public Distance<TF3, TF> {
 public:
  BoxShellDistance(TF3 inner_widths, TF thickness, TF outset_arg = 0)
      : half_inner_widths(inner_widths * TF{0.5}),
        half_outer_widths(inner_widths * TF{0.5} + thickness),
        outset(outset_arg),
        Distance<TF3, TF>(
            inner_widths * TF{-0.5} - thickness - outset_arg,
            inner_widths * TF{0.5} + thickness + outset_arg,
            length(inner_widths * TF{0.5} + thickness) + outset_arg) {}
  TF signedDistance(dg::Vector3r<TF> const& x) const override {
    TF3 inner_diff = TF3{abs(x(0)), abs(x(1)), abs(x(2))} - half_inner_widths;
    TF3 outer_diff = TF3{abs(x(0)), abs(x(1)), abs(x(2))} - half_outer_widths;
    TF3 inner_clipped_diff =
        TF3{max(inner_diff.x, TF{0}), max(inner_diff.y, TF{0}),
            max(inner_diff.z, TF{0})};
    TF3 outer_clipped_diff =
        TF3{max(outer_diff.x, TF{0}), max(outer_diff.y, TF{0}),
            max(outer_diff.z, TF{0})};
    TF inner_dist =
        length(inner_clipped_diff) +
        min(max(inner_diff.x, max(inner_diff.y, inner_diff.z)), TF{0});
    TF outer_dist =
        length(outer_clipped_diff) +
        min(max(outer_diff.x, max(outer_diff.y, outer_diff.z)), TF{0});
    return max(outer_dist, -inner_dist);
  }
  __device__ TF signed_distance(TF3 const& x) const {
    TF3 inner_diff = fabs(x) - half_inner_widths;
    TF3 outer_diff = fabs(x) - half_outer_widths;
    TF3 inner_clipped_diff = fmax(inner_diff, TF3{0});
    TF3 outer_clipped_diff = fmax(outer_diff, TF3{0});
    TF inner_dist =
        length(inner_clipped_diff) +
        min(max(inner_diff.x, max(inner_diff.y, inner_diff.z)), TF{0}) - outset;
    TF outer_dist =
        length(outer_clipped_diff) +
        min(max(outer_diff.x, max(outer_diff.y, outer_diff.z)), TF{0}) - outset;
    return max(outer_dist, -inner_dist);
  }
  // finite difference gives more stable results
  __device__ TF3 gradient(TF3 const& x, TF scale) const {
    constexpr TF kEps = 0.00390625;
    TF step = scale * kEps;
    return TF3{signed_distance(x + TF3{step, 0, 0}) -
                   signed_distance(x - TF3{step, 0, 0}),
               signed_distance(x + TF3{0, step, 0}) -
                   signed_distance(x - TF3{0, step, 0}),
               signed_distance(x + TF3{0, 0, step}) -
                   signed_distance(x - TF3{0, 0, step})};
  }
  TF3 half_inner_widths;
  TF3 half_outer_widths;
  TF outset;
};
}  // namespace dg
}  // namespace alluvion
#endif /* ALLUVION_BOX_SHELL_DISTANCE_HPP */
