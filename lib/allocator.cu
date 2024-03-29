#include <cuda_runtime.h>

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

void Allocator::copy(void* dst, void const* src, unsigned int num_bytes) {
  abort_if_error(cudaMemcpy(dst, src, num_bytes, cudaMemcpyDefault));
}

void Allocator::set(void* dst, unsigned int num_bytes, int value) {
  abort_if_error(cudaMemset(dst, value, num_bytes));
}

void Allocator::get_device_properties(cudaDeviceProp* prop, int device) {
  abort_if_error(cudaGetDeviceProperties(prop, device));
}

void Allocator::abort_if_error(cudaError_t err) {
  if (err != cudaSuccess) {
    const char* error_string = cudaGetErrorString(err);
    std::cerr << "CUDA API returns the error: " << error_string << std::endl;
    throw std::runtime_error(error_string);
  }
};
}  // namespace alluvion
