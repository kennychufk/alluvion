#ifndef ALLUVION_VARIABLE_HPP
#define ALLUVION_VARIABLE_HPP

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include "alluvion/allocator.hpp"
#include "alluvion/base_variable.hpp"
#include "alluvion/data_type.hpp"

namespace alluvion {
template <U D, typename M>
class Variable : public BaseVariable {
 public:
  Variable() : ptr_(nullptr) {}
  Variable(const Variable& var) = default;
  Variable(std::array<U, D> const& shape) : ptr_(nullptr) {
    for (U i = 0; i < D; ++i) {
      shape_[i] = shape[i];
    }
    Allocator::allocate<M>(&ptr_, get_num_elements());
    // std::cout << "constructed Variable with size " << shape_[0] << ", "
    //           << shape_[1] << ", " << shape_[2] << " at " << ptr_ <<
    //           std::endl;
  }
  virtual ~Variable() {}
  virtual void set_pointer(void* ptr) override { ptr_ = ptr; }
  constexpr NumericType get_type() const {
    if (typeid(M) == typeid(F) || typeid(M) == typeid(F2) ||
        typeid(M) == typeid(F3) || typeid(M) == typeid(F4))
      return typeid(F) == typeid(float) ? NumericType::f32 : NumericType::f64;
    if (typeid(M) == typeid(I)) return NumericType::i32;
    if (typeid(M) == typeid(U)) return NumericType::u32;
    return NumericType::undefined;
  }
  constexpr U get_vector_size() const {
    if (typeid(M) == typeid(F)) return 1;
    if (typeid(M) == typeid(I)) return 1;
    if (typeid(M) == typeid(U)) return 1;
    if (typeid(M) == typeid(F2)) return 2;
    if (typeid(M) == typeid(I2)) return 2;
    if (typeid(M) == typeid(U2)) return 2;
    if (typeid(M) == typeid(F3)) return 3;
    if (typeid(M) == typeid(I3)) return 3;
    if (typeid(M) == typeid(U3)) return 3;
    if (typeid(M) == typeid(F4)) return 4;
    if (typeid(M) == typeid(I4)) return 4;
    if (typeid(M) == typeid(U4)) return 4;
    return 0;
  }
  U get_num_vectors() const {
    return std::accumulate(std::begin(shape_), std::end(shape_), 1,
                           std::multiplies<U>());
  }
  U get_num_elements() const { return get_num_vectors() * get_vector_size(); }
  U get_num_bytes() const { return get_num_vectors() * sizeof(M); }
  void get_bytes(void* dst, U num_bytes) const {
    if (num_bytes == 0) return;
    if (num_bytes > get_num_bytes()) {
      std::cerr << "retrieving more than allocated" << std::endl;
      abort();
    }
    Allocator::copy_to_host(dst, ptr_, num_bytes);
  }
  void get_bytes(void* dst) const { get_bytes(dst, get_num_bytes()); }
  void set_bytes(void const* src, U num_bytes) {
    if (num_bytes == 0) return;
    if (num_bytes > get_num_bytes()) {
      std::cerr << "setting more than allocated: " << num_bytes << " "
                << get_num_bytes() << std::endl;
      abort();
    }
    Allocator::copy_to_device(ptr_, src, num_bytes);
  }
  void set_bytes(void const* src) { set_bytes(src, get_num_bytes()); }
  void set_zero() { Allocator::set_device(ptr_, get_num_bytes()); }

  constexpr __device__ M& operator()(U i) {
    return *(static_cast<M*>(ptr_) + i);
  }

  constexpr __device__ M& operator()(U i, U j) {
    return (D == 2) ? *(static_cast<M*>(ptr_) + (i * shape_[1] + j))
                    : *(static_cast<M*>(ptr_ - 0xffffffff));
  }

  constexpr __device__ M& operator()(U i, U j, U k) {
    return (D == 3) ? *(static_cast<M*>(ptr_) +
                        ((i * shape_[1] + j) * shape_[2] + k))
                    : *(static_cast<M*>(ptr_ - 0xffffffff));
  }

  constexpr __device__ M& operator()(I3 index) {
    return (D == 3) ? *(static_cast<M*>(ptr_) +
                        ((index.x * shape_[1] + index.y) * shape_[2] + index.z))
                    : *(static_cast<M*>(ptr_ - 0xffffffff));
  }

  constexpr __device__ M& operator()(U i, U j, U k, U l) {
    return (D == 4) ? *(static_cast<M*>(ptr_) +
                        (((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l))
                    : *(static_cast<M*>(ptr_ - 0xffffffff));
  }

  constexpr __device__ M& operator()(U4 index) {
    return (D == 4)
               ? *(static_cast<M*>(ptr_) +
                   (((index.x * shape_[1] + index.y) * shape_[2] + index.z) *
                        shape_[3] +
                    index.w))
               : *(static_cast<M*>(ptr_ - 0xffffffff));
  }

  constexpr __device__ M& operator()(I3 index, U l) {
    return (D == 4)
               ? *(static_cast<M*>(ptr_) +
                   (((index.x * shape_[1] + index.y) * shape_[2] + index.z) *
                        shape_[3] +
                    l))
               : *(static_cast<M*>(ptr_ - 0xffffffff));
  }

  U shape_[D];
  void* ptr_;
};

}  // namespace alluvion

#endif /* ALLUVION_VARIABLE_HPP */
