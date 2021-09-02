#ifndef ALLUVION_UNIQUE_DEVICE_POINTER_HPP
#define ALLUVION_UNIQUE_DEVICE_POINTER_HPP

#include <cuda_runtime.h>

namespace alluvion {
class UniqueDevicePointer {
 public:
  UniqueDevicePointer() = delete;
  UniqueDevicePointer(void* ptr, cudaTextureObject_t tex);
  UniqueDevicePointer(const UniqueDevicePointer&) = delete;
  virtual ~UniqueDevicePointer();
  void* ptr_;
  cudaTextureObject_t tex_;
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_DEVICE_POINTER_HPP */
