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
template <unsigned int D, typename M>
class Variable : public BaseVariable {
 public:
  Variable() : ptr_(nullptr) {}
  Variable(const Variable& var) = default;
  Variable(std::array<unsigned int, D> const& shape)
      : shape_(shape), ptr_(nullptr) {
    Allocator::allocate<M>(&ptr_, get_num_elements());
    // std::cout << "constructed Variable with size " << shape_[0] << ", "
    //           << shape_[1] << ", " << shape_[2] << " at " << ptr_ <<
    //           std::endl;
  }
  virtual ~Variable() {
    std::cout << "destructing Variable with addr " << ptr_ << std::endl;
  }
  virtual void set_pointer(void* ptr) override { ptr_ = ptr; }
  constexpr NumericType get_type() const {
    if (typeid(M) == typeid(F) || typeid(M) == typeid(F2) ||
        typeid(M) == typeid(F3) || typeid(M) == typeid(F4))
      return typeid(F) == typeid(float) ? NumericType::f32 : NumericType::f64;
    if (typeid(M) == typeid(I)) return NumericType::i32;
    if (typeid(M) == typeid(U)) return NumericType::u32;
    return NumericType::undefined;
  }
  constexpr unsigned int get_vector_size() const {
    if (typeid(M) == typeid(F) || typeid(M) == typeid(I) ||
        typeid(M) == typeid(U))
      return 1;
    if (typeid(M) == typeid(F2)) return 2;
    if (typeid(M) == typeid(F3)) return 3;
    if (typeid(M) == typeid(F4)) return 4;
    return 0;
  }
  unsigned int get_num_vectors() const {
    return std::accumulate(std::begin(shape_), std::end(shape_), 1,
                           std::multiplies<unsigned int>());
  }
  unsigned int get_num_elements() const {
    return get_num_vectors() * get_vector_size();
  }
  unsigned int get_num_bytes() const { return get_num_vectors() * sizeof(M); }
  void get_bytes(void* dst, unsigned int num_bytes) const {
    if (num_bytes == 0) return;
    if (num_bytes > get_num_bytes()) {
      std::cerr << "retrieving more than allocated" << std::endl;
      abort();
    }
    Allocator::copy_to_host(dst, ptr_, num_bytes);
  }
  void set_bytes(void const* src, unsigned int num_bytes) {
    if (num_bytes == 0) return;
    if (num_bytes > get_num_bytes()) {
      std::cerr << "setting more than allocated: " << num_bytes << " "
                << get_num_bytes() << std::endl;
      abort();
    }
    Allocator::copy_to_device(ptr_, src, num_bytes);
  }

  constexpr __device__ M& operator()(U i) {
    return (D == 1) ? *(reinterpret_cast<M*>(ptr_) + i)
                    : *(reinterpret_cast<M*>(0));
  }

  constexpr __device__ M& operator()(U i, U j) {
    return (D == 2) ? *(reinterpret_cast<M*>(ptr_) + i * shape_[1] + j)
                    : *(reinterpret_cast<M*>(0));
  }

  std::array<unsigned int, D> shape_;
  void* ptr_;
};

}  // namespace alluvion

#endif /* ALLUVION_VARIABLE_HPP */
