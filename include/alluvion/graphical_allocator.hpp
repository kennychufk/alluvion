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
    std::cout << "glGenBuffers" << std::endl;
    glGenBuffers(1, vbo);
    std::cout << "glBindBuffer" << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    std::cout << "glBufferData" << std::endl;
    glBufferData(GL_ARRAY_BUFFER, num_elements * sizeof(M), nullptr,
                 GL_DYNAMIC_DRAW);
    std::cout << "glBindBuffer" << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    std::cout << "cudaGraphicsRegisterFlagsNone" << std::endl;
    Allocator::abort_if_error(
        cudaGraphicsGLRegisterBuffer(res, *vbo, cudaGraphicsRegisterFlagsNone));
  };

  static void map(std::vector<cudaGraphicsResource*>& resources);
  static void* get_mapped_pointer(cudaGraphicsResource* res);
  static void unmap(std::vector<cudaGraphicsResource*>& resources);
  static void free(GLuint* vbo, cudaGraphicsResource** res);
};
}  // namespace alluvion

#endif /* ALLUVION_GRAPHICAL_ALLOCATOR_HPP */
