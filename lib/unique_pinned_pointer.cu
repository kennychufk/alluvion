#include <functional>
#include <iostream>
#include <numeric>

#include "alluvion/allocator.hpp"
#include "alluvion/unique_pinned_pointer.hpp"
namespace alluvion {
UniquePinnedPointer::UniquePinnedPointer(void* ptr) : ptr_(ptr){};
UniquePinnedPointer::~UniquePinnedPointer() {
  std::cout << "destructing UniquePinnedPointer with addr " << ptr_
            << std::endl;
  Allocator::free_pinned(&ptr_);
}
}  // namespace alluvion
