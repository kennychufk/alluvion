#include "alluvion/mesh_buffer.hpp"

namespace alluvion {
MeshBuffer::MeshBuffer()
    : vertex(0), index(0), num_vertices(0), num_indices(0) {}
void MeshBuffer::set_vertices(void const* src) {
  glBindBuffer(GL_ARRAY_BUFFER, vertex);
  glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices * sizeof(float3), src);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}
void MeshBuffer::set_indices(void const* src) {
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index);
  glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, num_indices * sizeof(U), src);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

}  // namespace alluvion
