#include <cuda_runtime.h>

#include <cstring>
#include <iostream>

#include "alluvion/allocator.hpp"

namespace alluvion {
Allocator::Allocator() {}
Allocator::~Allocator() {}

void Allocator::free(void** ptr) {
  if (ptr == nullptr) return;
  abort_if_error(cudaFree(*ptr));
  *ptr = nullptr;
}

void Allocator::free_pinned(void** ptr) {
  if (ptr == nullptr) return;
  abort_if_error(cudaFreeHost(*ptr));
  *ptr = nullptr;
}

cudaTextureObject_t Allocator::create_texture(
    void* ptr, size_t num_bytes, cudaChannelFormatDesc const& channel_desc) {
  cudaTextureObject_t tex = 0;
  if (num_bytes == 0) {
    return tex;
  }

  cudaResourceDesc res_desc;
  std::memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeLinear;
  res_desc.res.linear.devPtr = ptr;
  res_desc.res.linear.desc = channel_desc;
  if (res_desc.res.linear.desc.f == cudaChannelFormatKindNone) {
    return tex;
  }
  res_desc.res.linear.sizeInBytes = num_bytes;

  cudaTextureDesc tex_desc;
  std::memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = cudaReadModeElementType;

  abort_if_error(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));
  return tex;
}

void Allocator::destroy_texture(cudaTextureObject_t* tex) {
  if (*tex == 0) return;
  abort_if_error(cudaDestroyTextureObject(*tex));
  *tex = 0;
}

void Allocator::copy(void* dst, void const* src, size_t num_bytes) {
  abort_if_error(cudaMemcpy(dst, src, num_bytes, cudaMemcpyDefault));
}

void Allocator::set(void* dst, size_t num_bytes, int value) {
  abort_if_error(cudaMemset(dst, value, num_bytes));
}

void Allocator::get_device_properties(cudaDeviceProp* prop, int device) {
  abort_if_error(cudaGetDeviceProperties(prop, device));
}

void Allocator::abort_if_error(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA API returns the error: " << cudaGetErrorString(err)
              << std::endl;
    abort();
  }
};
}  // namespace alluvion
