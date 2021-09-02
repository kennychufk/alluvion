#include "alluvion/allocator.hpp"
#include "alluvion/graphical_allocator.hpp"
#include "alluvion/unique_graphical_resource.hpp"

namespace alluvion {
UniqueGraphicalResource::UniqueGraphicalResource(
    GLuint vbo, cudaGraphicsResource* res, void** mapped_ptr,
    cudaTextureObject_t* tex, U num_bytes, cudaChannelFormatDesc const& desc)
    : vbo_(vbo),
      res_(res),
      mapped_ptr_(mapped_ptr),
      tex_(tex),
      num_bytes_(num_bytes),
      desc_(desc) {}
UniqueGraphicalResource::~UniqueGraphicalResource() {
  GraphicalAllocator::free(&vbo_, &res_);
}
void UniqueGraphicalResource::set_mapped_pointer(void* ptr) {
  *mapped_ptr_ = ptr;
}
void UniqueGraphicalResource::create_texture() {
  *tex_ = Allocator::create_texture(*mapped_ptr_, num_bytes_, desc_);
}
void UniqueGraphicalResource::destroy_texture() {
  Allocator::destroy_texture(tex_);
}
}  // namespace alluvion
