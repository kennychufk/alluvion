#include "alluvion/mesh_buffer.hpp"

namespace alluvion {
MeshBuffer::MeshBuffer()
    : vertex(0), normal(0), texcoord(0), index(0), num_indices(0) {}
MeshBuffer::MeshBuffer(GLuint vertex_arg, GLuint normal_arg,
                       GLuint texcoord_arg, GLuint index_arg,
                       GLuint num_indices_arg)
    : vertex(vertex_arg),
      normal(normal_arg),
      texcoord(texcoord_arg),
      index(index_arg),
      num_indices(num_indices_arg) {}
}  // namespace alluvion
