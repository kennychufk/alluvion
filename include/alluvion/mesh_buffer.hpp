#ifndef ALLUVION_MESH_BUFFER_HPP
#define ALLUVION_MESH_BUFFER_HPP

#include <glad/glad.h>
//
#include "alluvion/data_type.hpp"

namespace alluvion {
struct MeshBuffer {
  GLuint vertex;
  GLuint index;
  U num_vertices;
  U num_indices;
  MeshBuffer();
  void set_vertices(void const* src);
  void set_indices(void const* src);
};
}  // namespace alluvion

#endif /* ALLUVION_MESH_BUFFER_HPP */
