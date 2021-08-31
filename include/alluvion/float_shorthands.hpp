#ifndef ALLUVION_FLOAT_SHORTHANDS_H
#define ALLUVION_ALLUVION_FLOAT_SHORTHANDS_H

#include <glad/glad.h>
// glad first
#include "alluvion/data_type.hpp"

namespace alluvion {
using F = float;
using F2 = float2;
using F3 = float3a;
using F4 = float4;
using Q = float4;
__device__ __host__ constexpr F operator"" _F(long double a) {
  return static_cast<F>(a);
}
constexpr GLenum GL_F = GL_FLOAT;
}  // namespace alluvion
#endif /* ALLUVION_ALLUVION_FLOAT_SHORTHANDS_H */
