#ifndef ALLUVION_UNIQUE_GRAPHICAL_RESOURCE_HPP
#define ALLUVION_UNIQUE_GRAPHICAL_RESOURCE_HPP

#include <glad/glad.h>

#include <alluvion/base_variable.hpp>
namespace alluvion {
class UniqueGraphicalResource {
 public:
  UniqueGraphicalResource() = delete;
  UniqueGraphicalResource(GLuint vbo, cudaGraphicsResource* res,
                          BaseVariable* var);
  UniqueGraphicalResource(const UniqueGraphicalResource&) = delete;
  virtual ~UniqueGraphicalResource();
  GLuint vbo_;
  cudaGraphicsResource* res_;
  BaseVariable* var_;
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_GRAPHICAL_RESOURCE_HPP */
