#include <cuda_runtime.h>

#include <iostream>

#include "alluvion/allocator.hpp"

namespace alluvion {
Allocator::Allocator() {}
Allocator::~Allocator() {}

void Allocator::free(void** ptr) {
  abort_if_error(cudaFree(*ptr));
  *ptr = nullptr;
}

void Allocator::copy_to_host(void* dst, void const* src,
                             unsigned int num_bytes) {
  // TODO: use cudaMemcpyDefault
  abort_if_error(cudaMemcpy(dst, src, num_bytes, cudaMemcpyDeviceToHost));
}

void Allocator::copy_to_device(void* dst, void const* src,
                               unsigned int num_bytes) {
  abort_if_error(cudaMemcpy(dst, src, num_bytes, cudaMemcpyHostToDevice));
}

void Allocator::set_device(void* dst, unsigned int num_bytes, int value) {
  abort_if_error(cudaMemset(dst, value, num_bytes));
}

void Allocator::abort_if_error(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA API returns the error: " << cudaGetErrorString(err)
              << std::endl;
    abort();
  }
};
}  // namespace alluvion
