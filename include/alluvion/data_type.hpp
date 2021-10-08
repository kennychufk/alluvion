#ifndef ALLUVION_DATA_TYPE_HPP
#define ALLUVION_DATA_TYPE_HPP

#include <typeinfo>
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

template <typename M>
NumericType get_numeric_type() {
  if (typeid(M) == typeid(float) || typeid(M) == typeid(float2) ||
      typeid(M) == typeid(float3) || typeid(M) == typeid(float4))
    return NumericType::f32;
  if (typeid(M) == typeid(double) || typeid(M) == typeid(double2) ||
      typeid(M) == typeid(double3) || typeid(M) == typeid(double4))
    return NumericType::f64;
  if (typeid(M) == typeid(I)) return NumericType::i32;
  if (typeid(M) == typeid(U)) return NumericType::u32;
  return NumericType::undefined;
}
template <typename M>
U get_num_primitives_for_numeric_type() {
  if (typeid(M) == typeid(float)) return 1;
  if (typeid(M) == typeid(double)) return 1;
  if (typeid(M) == typeid(I)) return 1;
  if (typeid(M) == typeid(U)) return 1;
  if (typeid(M) == typeid(float2)) return 2;
  if (typeid(M) == typeid(double2)) return 2;
  if (typeid(M) == typeid(I2)) return 2;
  if (typeid(M) == typeid(U2)) return 2;
  if (typeid(M) == typeid(float3)) return 3;
  if (typeid(M) == typeid(double3)) return 3;
  if (typeid(M) == typeid(I3)) return 3;
  if (typeid(M) == typeid(U3)) return 3;
  if (typeid(M) == typeid(float4)) return 4;
  if (typeid(M) == typeid(double4)) return 4;
  if (typeid(M) == typeid(I4)) return 4;
  if (typeid(M) == typeid(U4)) return 4;
  return 0;
}
}  // namespace alluvion

#endif
