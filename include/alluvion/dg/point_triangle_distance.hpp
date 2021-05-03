#ifndef ALLUVION_DG_POINT_TRIANGLE_DISTANCE_HPP
#define ALLUVION_DG_POINT_TRIANGLE_DISTANCE_HPP

#include <Eigen/Core>
#include <array>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {

enum class NearestEntity { VN0, VN1, VN2, EN0, EN1, EN2, FN };

F point_triangle_sqdistance(Vector3r const& point,
                            std::array<Vector3r const*, 3> const& triangle,
                            Vector3r* nearest_point = nullptr,
                            NearestEntity* ne = nullptr);

}  // namespace dg
}  // namespace alluvion
#endif
