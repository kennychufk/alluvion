#ifndef ALLUVION_DATA_TYPE_HPP
#define ALLUVION_DATA_TYPE_HPP

namespace alluvion {
using F = float;
using F2 = float2;
using F3 = float3;
using F4 = float4;

// using F = double;
// using F2 = double2;
// using F3 = double3;
// using F4 = double4;

using I = int;
using U = unsigned int;
enum class NumericType { f32, f64, i32, u32, undefined };

}  // namespace alluvion

#endif
