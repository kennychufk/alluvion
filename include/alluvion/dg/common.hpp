#ifndef ALLUVION_DG_COMMON_HPP
#define ALLUVION_DG_COMMON_HPP

#include <Eigen/Dense>

namespace alluvion {
namespace dg {
template <typename TF>
using Vector3r = Eigen::Matrix<TF, 3, 1, Eigen::DontAlign>;
template <typename TF>
using Matrix3r = Eigen::Matrix<TF, 3, 3, Eigen::DontAlign>;
template <typename TF>
using AlignedBox3r = Eigen::AlignedBox<TF, 3>;
}  // namespace dg
}  // namespace alluvion

#endif
