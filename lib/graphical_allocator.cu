#include "alluvion/graphical_allocator.hpp"

namespace alluvion {
GraphicalAllocator::GraphicalAllocator() {}
GraphicalAllocator::~GraphicalAllocator() {}

void GraphicalAllocator::free(GLuint* vbo, cudaGraphicsResource** res) {
  if (*res == nullptr) return;
  Allocator::abort_if_error(cudaGraphicsUnregisterResource(*res));
  *res = nullptr;
  if (*vbo == 0) return;
  glDeleteBuffers(1, vbo);
  *vbo = 0;
}

void GraphicalAllocator::map(std::vector<cudaGraphicsResource*>& resources) {
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
  Allocator::abort_if_error(
      cudaGraphicsUnmapResources(resources.size(), resources.data()));
}

}  // namespace alluvion
