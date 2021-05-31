#include "alluvion/graphical_allocator.hpp"
#include "alluvion/unique_texture.hpp"
namespace alluvion {
UniqueTexture::UniqueTexture(GLuint tex) : tex_(tex) {}
UniqueTexture::~UniqueTexture() { GraphicalAllocator::free_texture(&tex_); }
}  // namespace alluvion
