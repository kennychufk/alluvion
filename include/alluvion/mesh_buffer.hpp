#ifndef ALLUVION_MESH_BUFFER_HPP
#define ALLUVION_MESH_BUFFER_HPP

#include <glad/glad.h>
//
#include "alluvion/data_type.hpp"

namespace alluvion {
struct MeshBuffer {
  GLuint vertex;
  GLuint normal;
  GLuint texcoord;
  GLuint index;
  U num_indices;
  MeshBuffer();
  MeshBuffer(GLuint vertex_arg, GLuint normal_arg, GLuint texcoord_arg,
             GLuint index_arg, GLuint num_indices_arg);
};
}  // namespace alluvion

#endif /* ALLUVION_MESH_BUFFER_HPP */
