#ifndef ALLUVION_DATA_TYPE_HPP
#define ALLUVION_DATA_TYPE_HPP

namespace alluvion {
using I = int;
using I2 = int2;
using I3 = int3;
using I4 = int4;

using U = unsigned int;
using U2 = uint2;
using U3 = uint3;
using U4 = uint4;
enum class NumericType { f32, f64, i32, u32, undefined };
}  // namespace alluvion

#endif
