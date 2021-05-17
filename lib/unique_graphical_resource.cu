#include "alluvion/graphical_allocator.hpp"
#include "alluvion/unique_graphical_resource.hpp"

namespace alluvion {

UniqueGraphicalResource::UniqueGraphicalResource(GLuint vbo,
                                                 cudaGraphicsResource* res,
                                                 BaseVariable* var)
    : vbo_(vbo), res_(res), var_(var) {}
UniqueGraphicalResource::~UniqueGraphicalResource() {
  GraphicalAllocator::free(&vbo_, &res_);
}
}  // namespace alluvion
