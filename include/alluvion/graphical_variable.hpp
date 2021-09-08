#ifndef ALLUVION_GRAPHICAL_VARIABLE_HPP
#define ALLUVION_GRAPHICAL_VARIABLE_HPP

#include "alluvion/graphical_allocator.hpp"
#include "alluvion/variable.hpp"
namespace alluvion {
template <U D, typename M>
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html
// "The value set in devPtr may change every time that resource is mapped."
class GraphicalVariable : public Variable<D, M> {
 public:
  using Base = Variable<D, M>;
  GraphicalVariable(std::array<U, D> const& shape) : vbo_(0), res_(nullptr) {
    std::copy(std::begin(shape), std::end(shape), std::begin(Base::shape_));
    GraphicalAllocator::allocate<M>(&vbo_, &res_, Base::get_linear_shape());
  }
  virtual ~GraphicalVariable() {}

  GLuint vbo_;
  cudaGraphicsResource* res_;
};

}  // namespace alluvion

#endif /* ALLUVION_GRAPHICAL_VARIABLE_HPP */
