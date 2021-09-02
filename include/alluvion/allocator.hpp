#ifndef ALLUVION_ALLOCATOR_HPP
#define ALLUVION_ALLOCATOR_HPP

#include <iostream>

#include "alluvion/data_type.hpp"
namespace alluvion {
template <typename M>
inline cudaChannelFormatDesc create_channel_format_desc() {
  return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
}
template <>
inline cudaChannelFormatDesc create_channel_format_desc<U>() {
  int e = static_cast<int>(sizeof(U)) * 8;
  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}
template <>
inline cudaChannelFormatDesc create_channel_format_desc<float>() {
  int e = static_cast<int>(sizeof(float)) * 8;
  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}
template <>
inline cudaChannelFormatDesc create_channel_format_desc<float3a>() {
  int e = static_cast<int>(sizeof(float)) * 8;
  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}
template <>
inline cudaChannelFormatDesc create_channel_format_desc<float4>() {
  int e = static_cast<int>(sizeof(float)) * 8;
  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}
template <>
inline cudaChannelFormatDesc create_channel_format_desc<double>() {
  int e = static_cast<int>(sizeof(int)) * 8;
  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}
template <>
inline cudaChannelFormatDesc create_channel_format_desc<double3a>() {
  int e = static_cast<int>(sizeof(int)) * 8;
  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}
template <>
inline cudaChannelFormatDesc create_channel_format_desc<double4>() {
  int e = static_cast<int>(sizeof(int)) * 8;
  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

class Allocator {
 public:
  Allocator();
  virtual ~Allocator();
  template <typename M>
  static void allocate(void** ptr, U num_elements) {
    if (*ptr != nullptr) {
      std::cerr << "[Allocator] Pointer is dirty" << std::endl;
      abort();
    }
    if (num_elements == 0) {
      return;
    }
    abort_if_error(cudaMalloc(ptr, num_elements * sizeof(M)));
  };
  template <typename M>
  static void allocate_pinned(void** ptr, U num_elements) {
    if (*ptr != nullptr) {
      std::cerr << "[Allocator] Pointer is dirty" << std::endl;
      abort();
    }
    if (num_elements == 0) {
      return;
    }
    abort_if_error(cudaMallocHost(ptr, num_elements * sizeof(M)));
  }

  static cudaTextureObject_t create_texture(
      void* ptr, size_t num_bytes, cudaChannelFormatDesc const& channel_desc);

  template <typename M>
  static cudaTextureObject_t create_texture(void* ptr, size_t num_bytes) {
    return create_texture(ptr, num_bytes, create_channel_format_desc<M>());
  }

  static void free(void**);
  static void free_pinned(void**);
  static void destroy_texture(cudaTextureObject_t* tex);
  static void copy(void* dst, void const* src, size_t num_bytes);
  static void set(void* dst, size_t num_bytes, int value = 0);
  static void get_device_properties(cudaDeviceProp* prop, int device);
  static void abort_if_error(cudaError_t err);
};
}  // namespace alluvion

#endif /* ALLUVION_ALLOCATOR_HPP */
