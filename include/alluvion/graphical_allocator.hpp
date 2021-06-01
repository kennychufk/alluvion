#ifndef ALLUVION_GRAPHICAL_ALLOCATOR_HPP
#define ALLUVION_GRAPHICAL_ALLOCATOR_HPP

#include <glad/glad.h>
//
#include <cuda_gl_interop.h>

#include <array>
#include <iostream>
#include <vector>

#include "alluvion/allocator.hpp"
namespace alluvion {
class GraphicalAllocator {
 public:
  GraphicalAllocator();
  virtual ~GraphicalAllocator();
  template <typename M>
  static void allocate(GLuint* vbo, cudaGraphicsResource** res,
                       unsigned int num_elements) {
    if (*vbo != 0) {
      std::cerr << "[GraphicalAllocator] Buffer object is dirty" << std::endl;
      abort();
    }
    if (*res != nullptr) {
      std::cerr << "[GraphicalAllocator] Resource is dirty" << std::endl;
      abort();
    }
    if (num_elements == 0) {
      return;
    }
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, num_elements * sizeof(M), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    Allocator::abort_if_error(
        cudaGraphicsGLRegisterBuffer(res, *vbo, cudaGraphicsRegisterFlagsNone));
  };
  template <typename M>
  static GLuint allocate_array_buffer(U num_elements, GLenum usage,
                                      void const* src) {
    if (num_elements == 0) {
      return 0;
    }
    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, num_elements * sizeof(M), src, usage);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return buffer;
  }
  template <typename M>
  static GLuint allocate_static_array_buffer(U num_elements, void const* src) {
    return allocate_array_buffer<M>(num_elements, GL_STATIC_DRAW, src);
  }
  template <typename M>
  static GLuint allocate_dynamic_array_buffer(U num_elements, void const* src) {
    return allocate_array_buffer<M>(num_elements, GL_DYNAMIC_DRAW, src);
  }
  template <typename M>
  static GLuint allocate_element_array_buffer(U num_elements, void const* src) {
    if (num_elements == 0) {
      return 0;
    }
    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_elements * sizeof(M), src,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    return buffer;
  }
  static GLuint allocate_texture1d(std::array<GLfloat, 3> const* texture_data,
                                   GLsizei width);
  static GLuint allocate_monochrome_texture2d(unsigned char const* texture_data,
                                              GLsizei width, GLsizei height);
  static GLuint allocate_texture2d(unsigned char const* texture_data,
                                   GLsizei width, GLsizei height);
  static GLuint allocate_render_buffer(GLsizei width, GLsizei height);
  static GLuint allocate_framebuffer();
  static GLuint allocate_vao();
  static void reallocate_render_buffer(GLuint rbo, GLsizei width,
                                       GLsizei height);
  static void reallocate_texture2d(GLuint tex,
                                   unsigned char const* texture_data,
                                   GLsizei width, GLsizei height);

  static void map(std::vector<cudaGraphicsResource*>& resources);
  static void* get_mapped_pointer(cudaGraphicsResource* res);
  static void unmap(std::vector<cudaGraphicsResource*>& resources);
  static void free(GLuint* vbo, cudaGraphicsResource** res);
  static void free_buffer(GLuint* vbo);
  static void free_texture(GLuint* tex);
  static void free_vao(GLuint* tex);
  static void free_render_buffer(GLuint* rbo);
  static void free_framebuffer(GLuint* fbo);
};
}  // namespace alluvion

#endif /* ALLUVION_GRAPHICAL_ALLOCATOR_HPP */
