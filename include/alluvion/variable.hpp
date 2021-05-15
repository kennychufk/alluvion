#ifndef ALLUVION_VARIABLE_HPP
#define ALLUVION_VARIABLE_HPP

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "alluvion/allocator.hpp"
#include "alluvion/base_variable.hpp"
#include "alluvion/contact.hpp"
#include "alluvion/data_type.hpp"

namespace alluvion {
template <U D, typename M>
class Variable : public BaseVariable {
 public:
  Variable() : ptr_(nullptr) {}
  Variable(const Variable& var) = default;
  Variable(std::array<U, D> const& shape) : ptr_(nullptr) {
    std::copy(std::begin(shape), std::end(shape), std::begin(shape_));
    Allocator::allocate<M>(&ptr_, get_linear_shape());
  }
  virtual ~Variable() {}
  virtual void set_pointer(void* ptr) override { ptr_ = ptr; }
  // ==== numpy-related functions
  constexpr NumericType get_type() const {
    if (typeid(M) == typeid(F) || typeid(M) == typeid(F2) ||
        typeid(M) == typeid(F3) || typeid(M) == typeid(F4))
      return typeid(F) == typeid(float) ? NumericType::f32 : NumericType::f64;
    if (typeid(M) == typeid(I)) return NumericType::i32;
    if (typeid(M) == typeid(U)) return NumericType::u32;
    return NumericType::undefined;
  }
  constexpr U get_num_primitives_per_unit() const {
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
    if (typeid(M) == typeid(Contact)) return 0;
    return 0;
  }
  U get_num_primitives() const {
    return get_linear_shape() * get_num_primitives_per_unit();
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
  void get_bytes(void* dst, U num_bytes) const {
    if (num_bytes == 0) return;
    if (num_bytes > get_num_bytes()) {
      std::cerr << "retrieving more than allocated" << std::endl;
      abort();
    }
    Allocator::copy(dst, ptr_, num_bytes);
  }
  void get_bytes(void* dst) const { get_bytes(dst, get_num_bytes()); }
  void set_bytes(void const* src, U num_bytes) {
    if (num_bytes == 0) return;
    if (num_bytes > get_num_bytes()) {
      std::cerr << "setting more than allocated: " << num_bytes << " "
                << get_num_bytes() << std::endl;
      abort();
    }
    Allocator::copy(ptr_, src, num_bytes);
  }
  void set_bytes(void const* src) { set_bytes(src, get_num_bytes()); }
  void set_zero() { Allocator::set(ptr_, get_num_bytes()); }

  void write_file(const char* filename, U shape_outermost = 0) const {
    std::ofstream stream(filename, std::ios::binary | std::ios::trunc);
    if (shape_outermost > shape_[0]) {
      std::cerr << "Writing more than the outermost shape (" << shape_outermost
                << ">" << shape_[0] << ")." << std::endl;
      abort();
    }
    if (shape_outermost == 0) shape_outermost = shape_[0];
    for (U i = 0; i < D; ++i) {
      stream.write(
          reinterpret_cast<const char*>(i == 0 ? &shape_outermost : &shape_[i]),
          sizeof(U));
    }
    const U kEndOfShape = 0;
    stream.write(reinterpret_cast<const char*>(&kEndOfShape), sizeof(U));
    U num_primitives_per_unit = get_num_primitives_per_unit();
    stream.write(reinterpret_cast<const char*>(&num_primitives_per_unit),
                 sizeof(U));
    char type_label = numeric_type_to_label(get_type());
    stream.write(reinterpret_cast<const char*>(&type_label), sizeof(char));
    U linear_shape = get_linear_shape() / shape_[0] * shape_outermost;
    U num_bytes = linear_shape * sizeof(M);
    std::vector<M> host_copy(linear_shape);
    get_bytes(host_copy.data(), num_bytes);
    stream.write(reinterpret_cast<const char*>(host_copy.data()), num_bytes);
  }

  U read_file(const char* filename) {
    std::ifstream stream(filename, std::ios::binary);
    U shape_outermost;
    for (U i = 0; i < D; ++i) {
      U shape_item;
      stream.read(reinterpret_cast<char*>(&shape_item), sizeof(U));
      if (shape_item == 0) {
        std::cerr << "Shape mismatch when reading " << filename << std::endl;
        abort();
      }
      if (i == 0) {
        shape_outermost = shape_item;
        if (shape_outermost > shape_[0]) {
          std::cerr << "Reading more than the outermost shape ("
                    << shape_outermost << ">" << shape_[0] << ")." << std::endl;
          abort();
        }
      } else if (shape_[i] != shape_item) {
        std::cerr << "Shape mistmatch for shape item " << i << "(" << shape_[i]
                  << "!=" << shape_item << ")." << std::endl;
        abort();
      }
    }
    U end_of_shape;
    stream.read(reinterpret_cast<char*>(&end_of_shape), sizeof(U));
    if (end_of_shape != 0) {
      std::cerr << "Dimension mismatch when reading " << filename << std::endl;
      abort();
    }
    U num_primitives_per_unit;
    stream.read(reinterpret_cast<char*>(&num_primitives_per_unit), sizeof(U));
    if (num_primitives_per_unit != get_num_primitives_per_unit()) {
      std::cerr << "Num primitives per unit mismatch when reading " << filename
                << "(" << num_primitives_per_unit
                << "!=" << get_num_primitives_per_unit() << ")." << std::endl;
      abort();
    }
    char type_label;
    stream.read(reinterpret_cast<char*>(&type_label), sizeof(char));
    if (type_label != numeric_type_to_label(get_type())) {
      std::cerr << "Data type mismatch when reading " << filename << std::endl;
      abort();
    }
    U linear_shape = get_linear_shape() / shape_[0] * shape_outermost;
    U num_bytes = linear_shape * sizeof(M);
    std::vector<M> host_buffer(linear_shape);
    stream.read(reinterpret_cast<char*>(host_buffer.data()), num_bytes);
    set_bytes(host_buffer.data(), num_bytes);
    return shape_outermost;
  }

  static char numeric_type_to_label(NumericType numeric_type) {
    if (numeric_type == NumericType::f32) return 'f';
    if (numeric_type == NumericType::f64) return 'd';
    if (numeric_type == NumericType::i32) return 'i';
    if (numeric_type == NumericType::u32) return 'u';
    return '!';
  }

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
