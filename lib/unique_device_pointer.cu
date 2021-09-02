#include <functional>
#include <iostream>
#include <numeric>

#include "alluvion/allocator.hpp"
#include "alluvion/unique_device_pointer.hpp"
namespace alluvion {
UniqueDevicePointer::UniqueDevicePointer(void* ptr) : ptr_(ptr){};
UniqueDevicePointer::~UniqueDevicePointer() { Allocator::free(&ptr_); }
}  // namespace alluvion
