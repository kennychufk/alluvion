#ifndef ALLUVION_MESH_HPP
#define ALLUVION_MESH_HPP

#include <vector>

#include "alluvion/data_type.hpp"
#include "alluvion/dg/triangle_mesh.hpp"

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

  template <typename TF>
  void copy_to(dg::TriangleMesh<TF>& triangle_mesh) {
    triangle_mesh.m_vertices.resize(vertices.size());
    triangle_mesh.m_faces.resize(faces.size());
    for (U i = 0; i < vertices.size(); ++i) {
      dg::Vector3r<TF>& dst_v = triangle_mesh.m_vertices[i];
      float3 const& src_v = vertices[i];
      dst_v(0) = static_cast<TF>(src_v.x);
      dst_v(1) = static_cast<TF>(src_v.y);
      dst_v(2) = static_cast<TF>(src_v.z);
    }
    for (U i = 0; i < faces.size(); ++i) {
      std::array<unsigned int, 3>& dst_f = triangle_mesh.m_faces[i];
      U3 const& src_f = faces[i];
      dst_f[0] = src_f.x;
      dst_f[1] = src_f.y;
      dst_f[2] = src_f.z;
    }
    triangle_mesh.construct();
  }
};
}  // namespace alluvion

#endif /* ALLUVION_MESH_HPP */
