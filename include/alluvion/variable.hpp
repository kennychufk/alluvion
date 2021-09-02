#ifndef ALLUVION_VARIABLE_HPP
#define ALLUVION_VARIABLE_HPP

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "alluvion/allocator.hpp"
#include "alluvion/contact.hpp"
#include "alluvion/data_type.hpp"
#include "alluvion/texture.hpp"

namespace alluvion {
template <U D, typename M>
class Variable {
 public:
  Variable() : ptr_(nullptr), tex_(0) {}
  Variable(const Variable& var) = default;
  Variable(std::array<U, D> const& shape) : ptr_(nullptr), tex_(0) {
    std::copy(std::begin(shape), std::end(shape), std::begin(shape_));
    Allocator::allocate<M>(&ptr_, get_linear_shape());
    tex_ = Allocator::create_texture<M>(ptr_, get_num_bytes());
  }
  virtual ~Variable() {}
  // ==== numpy-related functions
  constexpr NumericType get_type() const {
    if (typeid(M) == typeid(float) || typeid(M) == typeid(float2) ||
        typeid(M) == typeid(float3a) || typeid(M) == typeid(float3) ||
        typeid(M) == typeid(float4))
      return NumericType::f32;
    if (typeid(M) == typeid(double) || typeid(M) == typeid(double2) ||
        typeid(M) == typeid(double3a) || typeid(M) == typeid(double3) ||
        typeid(M) == typeid(double4))
      return NumericType::f64;
    if (typeid(M) == typeid(I)) return NumericType::i32;
    if (typeid(M) == typeid(U)) return NumericType::u32;
    return NumericType::undefined;
  }
  constexpr U get_num_primitives_per_element() const {
    if (typeid(M) == typeid(float)) return 1;
    if (typeid(M) == typeid(double)) return 1;
    if (typeid(M) == typeid(I)) return 1;
    if (typeid(M) == typeid(U)) return 1;
    if (typeid(M) == typeid(float2)) return 2;
    if (typeid(M) == typeid(double2)) return 2;
    if (typeid(M) == typeid(I2)) return 2;
    if (typeid(M) == typeid(U2)) return 2;
    if (typeid(M) == typeid(float3)) return 3;
    if (typeid(M) == typeid(double3)) return 3;
    if (typeid(M) == typeid(I3)) return 3;
    if (typeid(M) == typeid(U3)) return 3;
    if (typeid(M) == typeid(float3a)) return 4;
    if (typeid(M) == typeid(double3a)) return 4;
    if (typeid(M) == typeid(float4)) return 4;
    if (typeid(M) == typeid(double4)) return 4;
    if (typeid(M) == typeid(I4)) return 4;
    if (typeid(M) == typeid(U4)) return 4;
    return 0;
  }
  constexpr U get_padding_per_element() const {
    if (typeid(M) == typeid(float3a)) return 1;
    if (typeid(M) == typeid(double3a)) return 1;
    return 0;
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
  size_t get_num_bytes() const {
    return static_cast<size_t>(get_linear_shape()) * sizeof(M);
  }
  void get_bytes(void* dst, size_t num_bytes, size_t offset = 0) const {
    if (num_bytes == 0) return;
    size_t byte_offset = offset * sizeof(M);
    if (num_bytes + byte_offset > get_num_bytes()) {
      std::cerr << "retrieving more than allocated" << std::endl;
      abort();
    }
    Allocator::copy(dst, static_cast<char*>(ptr_) + byte_offset, num_bytes);
  }
  void get_bytes(void* dst) const { get_bytes(dst, get_num_bytes()); }
  void set_bytes(void const* src, size_t num_bytes, size_t offset = 0) {
    if (num_bytes == 0) return;
    size_t byte_offset = offset * sizeof(M);
    if (num_bytes + byte_offset > get_num_bytes()) {
      std::cerr << "setting more than allocated: " << (num_bytes + byte_offset)
                << " " << get_num_bytes() << std::endl;
      abort();
    }
    Allocator::copy(static_cast<char*>(ptr_) + byte_offset, src, num_bytes);
  }
  void set_bytes(void const* src) { set_bytes(src, get_num_bytes()); }
  void set_from(Variable<D, M> const& src, U num_elements, size_t offset = 0) {
    set_bytes(src.ptr_, static_cast<size_t>(num_elements) * sizeof(M),
              offset * sizeof(M));
  }
  void set_from(Variable<D, M> const& src) {
    set_from(src, src.get_linear_shape());
  }
  void set_zero() { Allocator::set(ptr_, get_num_bytes()); }
  void set_same(int value, U num_elements, size_t offset = 0) {
    size_t num_bytes = static_cast<size_t>(num_elements) * sizeof(M);
    if (num_bytes == 0) return;
    size_t byte_offset = offset * sizeof(M);
    if (num_bytes + byte_offset > get_num_bytes()) {
      std::cerr << "setting more than allocated: " << (num_bytes + byte_offset)
                << " " << get_num_bytes() << std::endl;
      abort();
    }
    Allocator::set(static_cast<char*>(ptr_) + byte_offset, num_bytes, value);
  }
  void set_same(int value) { set_same(value, get_linear_shape()); }
  void scale(M multiplier, U num_elements, size_t offset = 0) {
    using namespace thrust::placeholders;
    thrust::transform(
        thrust::device_ptr<M>(static_cast<M*>(ptr_)) + offset,
        thrust::device_ptr<M>(static_cast<M*>(ptr_)) + (offset + num_elements),
        thrust::device_ptr<M>(static_cast<M*>(ptr_)) + offset, multiplier * _1);
  }
  void scale(M multiplier) { scale(multiplier, get_linear_shape()); }

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
    U num_primitives_per_element = get_num_primitives_per_element();
    stream.write(reinterpret_cast<const char*>(&num_primitives_per_element),
                 sizeof(U));
    char type_label = numeric_type_to_label(get_type());
    stream.write(reinterpret_cast<const char*>(&type_label), sizeof(char));
    U linear_shape = get_linear_shape() / shape_[0] * shape_outermost;
    size_t num_bytes = static_cast<size_t>(linear_shape) * sizeof(M);
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
    U linear_shape = get_linear_shape() / shape_[0] * shape_outermost;
    U num_primitives_per_element;
    stream.read(reinterpret_cast<char*>(&num_primitives_per_element),
                sizeof(U));
    if (num_primitives_per_element != get_num_primitives_per_element()) {
      std::cerr << "Num primitives per unit mismatch when reading " << filename
                << "(" << num_primitives_per_element
                << "!=" << get_num_primitives_per_element() << ")."
                << std::endl;
      abort();
    }
    char type_label;
    stream.read(reinterpret_cast<char*>(&type_label), sizeof(char));
    size_t num_bytes = static_cast<size_t>(linear_shape) * sizeof(M);
    std::vector<M> host_buffer(linear_shape);

    if (type_label == numeric_type_to_label(get_type())) {
      stream.read(reinterpret_cast<char*>(host_buffer.data()), num_bytes);
    } else if (type_label == 'f' && numeric_type_to_label(get_type()) == 'd') {
      U num_primitives = linear_shape * num_primitives_per_element;
      U num_source_bytes = sizeof(float) * num_primitives;
      std::vector<float> source_buffer(num_primitives);
      stream.read(reinterpret_cast<char*>(source_buffer.data()),
                  num_source_bytes);
      double* host_buffer_primitive_pointer =
          reinterpret_cast<double*>(host_buffer.data());
      for (U i = 0; i < source_buffer.size(); ++i) {
        host_buffer_primitive_pointer[i] =
            static_cast<double>(source_buffer[i]);
      }
    } else if (type_label == 'd' && numeric_type_to_label(get_type()) == 'f') {
      U num_primitives = linear_shape * num_primitives_per_element;
      U num_source_bytes = sizeof(double) * num_primitives;
      std::vector<double> source_buffer(num_primitives);
      stream.read(reinterpret_cast<char*>(source_buffer.data()),
                  num_source_bytes);
      float* host_buffer_primitive_pointer =
          reinterpret_cast<float*>(host_buffer.data());
      for (U i = 0; i < source_buffer.size(); ++i) {
        host_buffer_primitive_pointer[i] = static_cast<float>(source_buffer[i]);
      }
    } else {
      std::cerr << "Data type mismatch when reading " << filename << std::endl;
      abort();
    }
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

  Texture<D, M> get_tex() const { return Texture<D, M>(tex_, shape_); }

  constexpr __device__ M& operator()(U i) {
    return *(static_cast<M*>(ptr_) + i);
  }

  constexpr __device__ M& operator()(U i, U j) {
    static_assert(D == 2);
    return *(static_cast<M*>(ptr_) + (i * shape_[1] + j));
  }

  constexpr __device__ M& operator()(U i, U j, U k) {
    static_assert(D == 3);
    return *(static_cast<M*>(ptr_) + ((i * shape_[1] + j) * shape_[2] + k));
  }

  constexpr __device__ M& operator()(I3 index) {
    static_assert(D == 3);
    return *(static_cast<M*>(ptr_) +
             ((index.x * shape_[1] + index.y) * shape_[2] + index.z));
  }

  constexpr __device__ M& operator()(U i, U j, U k, U l) {
    static_assert(D == 4);
    return *(static_cast<M*>(ptr_) +
             (((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l));
  }

  constexpr __device__ M& operator()(U4 index) {
    static_assert(D == 4);
    return *(
        static_cast<M*>(ptr_) +
        (((index.x * shape_[1] + index.y) * shape_[2] + index.z) * shape_[3] +
         index.w));
  }

  constexpr __device__ M& operator()(I3 index, U l) {
    static_assert(D == 4);
    return *(
        static_cast<M*>(ptr_) +
        (((index.x * shape_[1] + index.y) * shape_[2] + index.z) * shape_[3] +
         l));
  }

  U shape_[D];
  void* ptr_;
  cudaTextureObject_t tex_;
};

}  // namespace alluvion

#endif /* ALLUVION_VARIABLE_HPP */
