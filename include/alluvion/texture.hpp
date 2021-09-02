#ifndef ALLUVION_TEXTURE_HPP
#define ALLUVION_TEXTURE_HPP

#include <algorithm>
#include <cstring>

namespace alluvion {
template <U D, typename M>
struct Texture {
  Texture(cudaTextureObject_t tex_arg, const U shape_arg[D]) : tex(tex_arg) {
    std::copy(shape_arg, shape_arg + D, std::begin(shape));
    std::memcpy(shape, shape_arg, sizeof(shape));
  }

  constexpr __device__ M operator()(U i) {
    if constexpr (std::is_same_v<M, double>) {
      int2 result = tex1Dfetch<int2>(tex, i);
      return reinterpret_cast<double&>(result);
    } else if constexpr (std::is_same_v<M, float3a>) {
      float4 result = tex1Dfetch<float4>(tex, i);
      return reinterpret_cast<float3a&>(result);
    } else if constexpr (std::is_same_v<M, double3a>) {
      int4 result[2];
      result[0] = tex1Dfetch<int4>(tex, i * 2);
      result[1] = tex1Dfetch<int4>(tex, i * 2 + 1);
      return *reinterpret_cast<double3a*>(result);
    } else if constexpr (std::is_same_v<M, double4>) {
      int4 result[2];
      result[0] = tex1Dfetch<int4>(tex, i * 2);
      result[1] = tex1Dfetch<int4>(tex, i * 2 + 1);
      return *reinterpret_cast<double4*>(result);
    } else {
      return tex1Dfetch<M>(tex, i);
    }
  }

  constexpr __device__ M operator()(U i, U j) {
    static_assert(D == 2);
    return operator()(i* shape[1] + j);
  }

  constexpr __device__ M operator()(U i, U j, U k) {
    static_assert(D == 3);
    return operator()((i * shape[1] + j) * shape[2] + k);
  }

  constexpr __device__ M operator()(I3 index) {
    static_assert(D == 3);
    return operator()((index.x * shape[1] + index.y) * shape[2] + index.z);
  }

  constexpr __device__ M operator()(U i, U j, U k, U l) {
    static_assert(D == 4);
    return operator()(((i * shape[1] + j) * shape[2] + k) * shape[3] + l);
  }

  constexpr __device__ M operator()(U4 index) {
    static_assert(D == 4);
    return operator()(((index.x * shape[1] + index.y) * shape[2] + index.z) *
                          shape[3] +
                      index.w);
  }

  constexpr __device__ M operator()(I3 index, U l) {
    static_assert(D == 4);
    return operator()(static_cast<U>(
        ((index.x * shape[1] + index.y) * shape[2] + index.z) * shape[3] + l));
  }

  U shape[D];
  cudaTextureObject_t tex;
};
}  // namespace alluvion
#endif
