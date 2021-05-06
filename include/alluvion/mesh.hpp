#ifndef ALLUVION_MESH_HPP
#define ALLUVION_MESH_HPP

#include <vector>

#include "alluvion/data_type.hpp"

namespace alluvion {
using VertexList = std::vector<F3>;
using TexcoordList = std::vector<F2>;
using FaceList = std::vector<U3>;
struct Mesh {
  VertexList vertices;
  VertexList normals;
  TexcoordList texcoords;
  FaceList faces;

  Mesh();
  void set_uv_sphere(F radius, U num_sectors, U num_stacks);
  void set_obj(const char* filename);
  void clear();
};
}  // namespace alluvion

#endif /* ALLUVION_MESH_HPP */