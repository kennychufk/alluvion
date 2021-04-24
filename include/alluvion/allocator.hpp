#ifndef ALLUVION_ALLOCATOR_HPP
#define ALLUVION_ALLOCATOR_HPP

#include <iostream>

#include "alluvion/data_type.hpp"
namespace alluvion {
class Allocator {
 public:
  Allocator();
  virtual ~Allocator();
  template <typename M>
  static void allocate(void** ptr, unsigned int num_elements) {
    if (*ptr != nullptr) {
      std::cerr << "[Allocator] Pointer is dirty" << std::endl;
      abort();
    }
    if (num_elements == 0) {
      return;
    }
    abort_if_error(cudaMalloc(ptr, num_elements * sizeof(M)));
  };
  static void free(void**);
  static void copy_to_host(void* dst, void const* src, unsigned int num_bytes);
  static void copy_to_device(void* dst, void const* src,
                             unsigned int num_bytes);
  static void abort_if_error(cudaError_t err);
};
}  // namespace alluvion

#endif /* ALLUVION_ALLOCATOR_HPP */
