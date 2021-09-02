#ifndef ALLUVION_UNIQUE_GRAPHICAL_RESOURCE_HPP
#define ALLUVION_UNIQUE_GRAPHICAL_RESOURCE_HPP

#include <glad/glad.h>

namespace alluvion {
class UniqueGraphicalResource {
 public:
  UniqueGraphicalResource() = delete;
  UniqueGraphicalResource(GLuint vbo, cudaGraphicsResource* res,
                          void** mapped_ptr, cudaTextureObject_t* tex,
                          U num_bytes, cudaChannelFormatDesc const& desc);
  UniqueGraphicalResource(const UniqueGraphicalResource&) = delete;
  void set_mapped_pointer(void* ptr);
  void create_texture();
  void destroy_texture();
  virtual ~UniqueGraphicalResource();
  GLuint vbo_;
  cudaGraphicsResource* res_;
  void** mapped_ptr_;
  cudaTextureObject_t* tex_;
  // necessary for texture creation
  U num_bytes_;
  cudaChannelFormatDesc desc_;
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_GRAPHICAL_RESOURCE_HPP */
