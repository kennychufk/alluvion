#ifndef ALLUVION_UNIQUE_PINNED_POINTER_HPP
#define ALLUVION_UNIQUE_PINNED_POINTER_HPP

namespace alluvion {
class UniquePinnedPointer {
 public:
  UniquePinnedPointer() = delete;
  UniquePinnedPointer(void* ptr);
  UniquePinnedPointer(const UniquePinnedPointer&) = delete;
  virtual ~UniquePinnedPointer();
  void* ptr_;
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_PINNED_POINTER_HPP */
