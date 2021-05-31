#include "alluvion/graphical_allocator.hpp"
#include "alluvion/unique_vbo.hpp"
namespace alluvion {
UniqueVbo::UniqueVbo(GLuint vbo) : vbo_(vbo) {}
UniqueVbo::~UniqueVbo() { GraphicalAllocator::free_buffer(&vbo_); }
}  // namespace alluvion
