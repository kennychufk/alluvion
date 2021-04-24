#include "alluvion/graphical_allocator.hpp"
#include "alluvion/unique_graphical_resource.hpp"

namespace alluvion {

UniqueGraphicalResource::UniqueGraphicalResource(GLuint vbo,
                                                 cudaGraphicsResource* res,
                                                 BaseVariable* var)
    : vbo_(vbo), res_(res), var_(var) {}
UniqueGraphicalResource::~UniqueGraphicalResource() {
  std::cout << "destructing UniqueGraphicalResource with id " << vbo_
            << std::endl;
  std::cout << "free" << std::endl;
  GraphicalAllocator::free(&vbo_, &res_);
  std::cout << "freed" << std::endl;
}
}  // namespace alluvion
