#include "alluvion/graphical_allocator.hpp"

namespace alluvion {
GraphicalAllocator::GraphicalAllocator() {}
GraphicalAllocator::~GraphicalAllocator() {}

GLuint GraphicalAllocator::allocate_texture1d(
    std::array<GLfloat, 3> const* texture_data, GLsizei width) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_1D, tex);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, width, 0, GL_RGB, GL_FLOAT,
               texture_data);
  glBindTexture(GL_TEXTURE_1D, 0);
  return tex;
}

GLuint GraphicalAllocator::allocate_monochrome_texture2d(
    unsigned char const* texture_data, GLsizei width, GLsizei height) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED,
               GL_UNSIGNED_BYTE, texture_data);
  glBindTexture(GL_TEXTURE_2D, 0);
  return tex;
}

GLuint GraphicalAllocator::allocate_vao() {
  GLuint vao;
  glGenVertexArrays(1, &vao);
  return vao;
}

void GraphicalAllocator::free(GLuint* vbo, cudaGraphicsResource** res) {
  if (*res == nullptr) return;
  Allocator::abort_if_error(cudaGraphicsUnregisterResource(*res));
  *res = nullptr;
  if (*vbo == 0) return;
  glDeleteBuffers(1, vbo);
  *vbo = 0;
}

void GraphicalAllocator::free_buffer(GLuint* vbo) {
  if (*vbo == 0) return;
  glDeleteBuffers(1, vbo);
  *vbo = 0;
}

void GraphicalAllocator::free_texture(GLuint* tex) {
  if (*tex == 0) return;
  glDeleteTextures(1, tex);
  *tex = 0;
}

void GraphicalAllocator::free_vao(GLuint* vao) {
  if (*vao == 0) return;
  glDeleteVertexArrays(1, vao);
  *vao = 0;
}

void GraphicalAllocator::map(std::vector<cudaGraphicsResource*>& resources) {
  if (resources.size() == 0) return;
  Allocator::abort_if_error(
      cudaGraphicsMapResources(resources.size(), resources.data()));
}

void* GraphicalAllocator::get_mapped_pointer(cudaGraphicsResource* res) {
  void* ptr;
  std::size_t returned_buffer_size;
  Allocator::abort_if_error(
      cudaGraphicsResourceGetMappedPointer(&ptr, &returned_buffer_size, res));
  return ptr;
}

void GraphicalAllocator::unmap(std::vector<cudaGraphicsResource*>& resources) {
  if (resources.size() == 0) return;
  Allocator::abort_if_error(
      cudaGraphicsUnmapResources(resources.size(), resources.data()));
}

}  // namespace alluvion
