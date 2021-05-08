#ifndef ALLUVION_PINNED_VARIABLE_HPP
#define ALLUVION_PINNED_VARIABLE_HPP

#include "alluvion/base_variable.hpp"
namespace alluvion {
template <U D, typename M>
class PinnedVariable : public BaseVariable {
 public:
  PinnedVariable() : ptr_(nullptr) {}
  PinnedVariable(const PinnedVariable& var) = default;
  PinnedVariable(std::array<U, D> const& shape) : ptr_(nullptr) {
    std::copy(std::begin(shape), std::end(shape), std::begin(shape_));
    Allocator::allocate_pinned<M>(&ptr_, get_linear_shape());
  }
  virtual ~PinnedVariable() {}
  virtual void set_pointer(void* ptr) override { ptr_ = ptr; }
  U get_linear_shape() const {
    return std::accumulate(std::begin(shape_), std::end(shape_), 1,
                           std::multiplies<U>());
  }
  U get_num_bytes() const { return get_linear_shape() * sizeof(M); }
  void set_bytes(void const* src, U num_bytes) {
    if (num_bytes == 0) return;
    if (num_bytes > get_num_bytes()) {
      std::cerr << "setting more than allocated: " << num_bytes << " "
                << get_num_bytes() << std::endl;
      abort();
    }
    Allocator::copy(ptr_, src, num_bytes);
  }
  constexpr __host__ M& operator()(U i) { return *(static_cast<M*>(ptr_) + i); }
  constexpr __host__ M const& operator()(U i) const {
    return *(static_cast<M*>(ptr_) + i);
  }

  U shape_[D];
  void* ptr_;
};

}  // namespace alluvion

#endif /* ALLUVION_PINNED_VARIABLE_HPP */
