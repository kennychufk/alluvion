#ifndef ALLUVION_UNIQUE_DEVICE_POINTER_HPP
#define ALLUVION_UNIQUE_DEVICE_POINTER_HPP

namespace alluvion {
class UniqueDevicePointer {
 public:
  UniqueDevicePointer() = delete;
  UniqueDevicePointer(void* ptr);
  UniqueDevicePointer(const UniqueDevicePointer&) = delete;
  virtual ~UniqueDevicePointer();
  void* ptr_;
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_DEVICE_POINTER_HPP */
