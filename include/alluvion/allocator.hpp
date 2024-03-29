#ifndef ALLUVION_ALLOCATOR_HPP
#define ALLUVION_ALLOCATOR_HPP

#include <iostream>
#include <sstream>

#include "alluvion/data_type.hpp"
namespace alluvion {
class Allocator {
 public:
  Allocator();
  virtual ~Allocator();
  template <typename M>
  static void allocate(void** ptr, unsigned int num_elements) {
    if (*ptr != nullptr) {
      std::stringstream error_sstream;
      error_sstream << "[Allocator] Pointer is dirty" << std::endl;
      throw std::runtime_error(error_sstream.str());
    }
    if (num_elements == 0) {
      return;
    }
    abort_if_error(cudaMalloc(ptr, num_elements * sizeof(M)));
  };
  template <typename M>
  static void allocate_pinned(void** ptr, unsigned int num_elements) {
    if (*ptr != nullptr) {
      std::stringstream error_sstream;
      error_sstream << "[Allocator] Pointer is dirty" << std::endl;
      throw std::runtime_error(error_sstream.str());
    }
    if (num_elements == 0) {
      return;
    }
    abort_if_error(cudaMallocHost(ptr, num_elements * sizeof(M)));
  }
  static void free(void**);
  static void free_pinned(void**);
  static void copy(void* dst, void const* src, unsigned int num_bytes);
  static void set(void* dst, unsigned int num_bytes, int value = 0);
  static void get_device_properties(cudaDeviceProp* prop, int device);
  static void abort_if_error(cudaError_t err);
};
}  // namespace alluvion

#endif /* ALLUVION_ALLOCATOR_HPP */
