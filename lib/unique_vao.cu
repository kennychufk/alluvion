#include "alluvion/graphical_allocator.hpp"
#include "alluvion/unique_vao.hpp"
namespace alluvion {
UniqueVao::UniqueVao() : vao_(0) {}
UniqueVao::UniqueVao(GLuint vao) : vao_(vao) {}
UniqueVao::~UniqueVao() { GraphicalAllocator::free_vao(&vao_); }
void UniqueVao::set(GLuint vao) {
  GraphicalAllocator::free_vao(&vao_);
  vao_ = vao;
}
}  // namespace alluvion
