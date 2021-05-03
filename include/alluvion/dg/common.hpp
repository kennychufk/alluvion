#ifndef ALLUVION_DG_COMMON_HPP
#define ALLUVION_DG_COMMON_HPP

#include <Eigen/Dense>

#include "alluvion/data_type.hpp"

namespace alluvion {
namespace dg {
using Vector3r = Eigen::Matrix<F, 3, 1, Eigen::DontAlign>;
using Matrix3r = Eigen::Matrix<F, 3, 3, Eigen::DontAlign>;
using AlignedBox3r = Eigen::AlignedBox<F, 3>;
}  // namespace dg
}  // namespace alluvion

#endif
