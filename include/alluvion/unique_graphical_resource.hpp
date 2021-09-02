#ifndef ALLUVION_UNIQUE_GRAPHICAL_RESOURCE_HPP
#define ALLUVION_UNIQUE_GRAPHICAL_RESOURCE_HPP

#include <glad/glad.h>

namespace alluvion {
class UniqueGraphicalResource {
 public:
  UniqueGraphicalResource() = delete;
  UniqueGraphicalResource(GLuint vbo, cudaGraphicsResource* res,
                          void** mapped_ptr);
  UniqueGraphicalResource(const UniqueGraphicalResource&) = delete;
  void set_mapped_pointer(void* ptr);
  virtual ~UniqueGraphicalResource();
  GLuint vbo_;
  cudaGraphicsResource* res_;
  void** mapped_ptr_;
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_GRAPHICAL_RESOURCE_HPP */
