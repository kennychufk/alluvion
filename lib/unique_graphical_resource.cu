#include "alluvion/graphical_allocator.hpp"
#include "alluvion/unique_graphical_resource.hpp"

namespace alluvion {

UniqueGraphicalResource::UniqueGraphicalResource(GLuint vbo,
                                                 cudaGraphicsResource* res,
                                                 void** mapped_ptr)
    : vbo_(vbo), res_(res), mapped_ptr_(mapped_ptr) {}
UniqueGraphicalResource::~UniqueGraphicalResource() {
  GraphicalAllocator::free(&vbo_, &res_);
}
void UniqueGraphicalResource::set_mapped_pointer(void* ptr) {
  *mapped_ptr_ = ptr;
}
}  // namespace alluvion
