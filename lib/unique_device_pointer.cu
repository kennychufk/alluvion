#include <functional>
#include <iostream>
#include <numeric>

#include "alluvion/allocator.hpp"
#include "alluvion/unique_device_pointer.hpp"
namespace alluvion {
UniqueDevicePointer::UniqueDevicePointer(void* ptr) : ptr_(ptr){};
UniqueDevicePointer::~UniqueDevicePointer() {
  std::cout << "destructing UniqueDevicePointer with addr " << ptr_
            << std::endl;
  Allocator::free(&ptr_);
}
}  // namespace alluvion
