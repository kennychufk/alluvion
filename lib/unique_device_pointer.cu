#include <functional>
#include <iostream>
#include <numeric>

#include "alluvion/allocator.hpp"
#include "alluvion/unique_device_pointer.hpp"
namespace alluvion {
UniqueDevicePointer::UniqueDevicePointer(void* ptr, cudaTextureObject_t tex)
    : ptr_(ptr), tex_(tex){};
UniqueDevicePointer::~UniqueDevicePointer() {
  Allocator::destroy_texture(&tex_);
  Allocator::free(&ptr_);
}
}  // namespace alluvion
