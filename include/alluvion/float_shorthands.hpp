#ifndef ALLUVION_FLOAT_SHORTHANDS_H
#define ALLUVION_ALLUVION_FLOAT_SHORTHANDS_H
namespace alluvion {
using F = float;
using F2 = float2;
using F3 = float3;
using F4 = float4;
using Q = float4;
__device__ __host__ constexpr F operator"" _F(long double a) {
  return static_cast<F>(a);
}
}  // namespace alluvion
#endif /* ALLUVION_ALLUVION_FLOAT_SHORTHANDS_H */
