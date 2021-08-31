#ifndef ALLUVION_DOUBLE_SHORTHANDS_H
#define ALLUVION_ALLUVION_DOUBLE_SHORTHANDS_H

#include <glad/glad.h>
// glad first
#include "alluvion/data_type.hpp"

namespace alluvion {
using F = double;
using F2 = double2;
using F3 = double3a;
using F4 = double4;
using Q = double4;
__device__ __host__ constexpr F operator"" _F(long double a) {
  return static_cast<F>(a);
}
constexpr GLenum GL_F = GL_DOUBLE;
}  // namespace alluvion
#endif /* ALLUVION_ALLUVION_DOUBLE_SHORTHANDS_H */
