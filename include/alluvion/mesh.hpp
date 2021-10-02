#ifndef ALLUVION_MESH_HPP
#define ALLUVION_MESH_HPP

#include <vector>

#include "alluvion/data_type.hpp"

namespace alluvion {
using VertexList = std::vector<float3>;
using TexcoordList = std::vector<float2>;
using FaceList = std::vector<U3>;
struct Mesh {
  VertexList vertices;
  VertexList normals;
  TexcoordList texcoords;
  FaceList faces;

  Mesh();
  void set_box(float3 widths, U n = 2);
  void set_uv_sphere(float radius, U num_sectors, U num_stacks);
  void set_cylinder(float radius, float height, U num_sectors, U num_stacks);
  void set_obj(const char* filename);
  void export_obj(const char* filename);
  void calculate_normals();
  void translate(float3 dx);
  void rotate(float4 q);
  void scale(float s);
  void clear();
  float calculate_mass_properties(float3& com, float3& inertia_diag,
                                  float3& inertia_off_diag,
                                  float density = 1) const;
  static void face_integrals(float3 const* face_vertices[3],
                             float const& weight, float3 const& face_normal,
                             int a, int b, int c, float& Fa, float& Fb,
                             float& Fc, float& Faa, float& Fbb, float& Fcc,
                             float& Faaa, float& Fbbb, float& Fccc, float& Faab,
                             float& Fbbc, float& Fcca);
};
}  // namespace alluvion

#endif /* ALLUVION_MESH_HPP */
