#ifndef ALLUVION_PINNED_VARIABLE_HPP
#define ALLUVION_PINNED_VARIABLE_HPP

#include "alluvion/data_type.hpp"

namespace alluvion {
template <U D, typename M>
class PinnedVariable {
 public:
  PinnedVariable() : ptr_(nullptr) {}
  PinnedVariable(const PinnedVariable& var) = default;
  PinnedVariable(std::array<U, D> const& shape) : ptr_(nullptr) {
    std::copy(std::begin(shape), std::end(shape), std::begin(shape_));
    Allocator::allocate_pinned<M>(&ptr_, get_linear_shape());
  }
  virtual ~PinnedVariable() {}
  // ==== numpy-related functions
  constexpr NumericType get_type() const { return get_numeric_type<M>(); }
  constexpr U get_num_primitives_per_element() const {
    return get_num_primitives_for_numeric_type<M>();
  }
  U get_num_primitives() const {
    return get_linear_shape() * get_num_primitives_per_element();
  }
  std::array<U, D> get_shape() const {
    std::array<U, D> shape_std_array;
    std::copy(std::begin(shape_), std::end(shape_),
              std::begin(shape_std_array));
    return shape_std_array;
  }
  // ==== End of numpy-related functions
  U get_linear_shape() const {
    return std::accumulate(std::begin(shape_), std::end(shape_), 1,
                           std::multiplies<U>());
  }
  U get_num_bytes() const { return get_linear_shape() * sizeof(M); }
  void get_bytes(void* dst, U num_bytes, U offset = 0) const {
    if (num_bytes == 0) return;
    U byte_offset = offset * sizeof(M);
    if (num_bytes + byte_offset > get_num_bytes()) {
      std::stringstream error_sstream;
      error_sstream << "retrieving more than allocated" << std::endl;
      throw std::runtime_error(error_sstream.str());
    }
    Allocator::copy(dst, static_cast<char*>(ptr_) + byte_offset, num_bytes);
  }
  void get_bytes(void* dst) const { get_bytes(dst, get_num_bytes()); }
  void set_bytes(void const* src, U num_bytes, U offset = 0) {
    if (num_bytes == 0) return;
    U byte_offset = offset * sizeof(M);
    if (num_bytes + byte_offset > get_num_bytes()) {
      std::stringstream error_sstream;
      error_sstream << "setting more than allocated: "
                    << (num_bytes + byte_offset) << " " << get_num_bytes()
                    << std::endl;
      throw std::runtime_error(error_sstream.str());
    }
    Allocator::copy(static_cast<char*>(ptr_) + byte_offset, src, num_bytes);
  }
  void set_bytes(void const* src) { set_bytes(src, get_num_bytes()); }
  constexpr __host__ M& operator()(U i) { return *(static_cast<M*>(ptr_) + i); }
  constexpr __host__ M const& operator()(U i) const {
    return *(static_cast<M*>(ptr_) + i);
  }

  U shape_[D];
  void* ptr_;
};

}  // namespace alluvion

#endif /* ALLUVION_PINNED_VARIABLE_HPP */
