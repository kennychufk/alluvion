#ifndef ALLUVION_GRAPHICAL_ALLOCATOR_HPP
#define ALLUVION_GRAPHICAL_ALLOCATOR_HPP

#include <glad/glad.h>
//
#include <cuda_gl_interop.h>

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
  static GLuint allocate_static_array_buffer(U num_elements, void const* src) {
    if (num_elements == 0) {
      return 0;
    }
    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, num_elements * sizeof(M), src,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return buffer;
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

  static void map(std::vector<cudaGraphicsResource*>& resources);
  static void* get_mapped_pointer(cudaGraphicsResource* res);
  static void unmap(std::vector<cudaGraphicsResource*>& resources);
  static void free(GLuint* vbo, cudaGraphicsResource** res);
  static void free_buffer(GLuint* vbo);
};
}  // namespace alluvion

#endif /* ALLUVION_GRAPHICAL_ALLOCATOR_HPP */
