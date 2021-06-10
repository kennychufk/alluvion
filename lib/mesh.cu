#include <array>
#include <fstream>
#include <sstream>
#include <string>

#include "alluvion/constants.hpp"
#include "alluvion/mesh.hpp"
#include "alluvion/runner.hpp"
namespace alluvion {
Mesh::Mesh() {}
void Mesh::set_uv_sphere(float radius, U num_sectors, U num_stacks) {
  ///////////////////////////////////////////////////////////////////////////////
  // Sphere.cpp
  // ==========
  // Sphere for OpenGL with (radius, sectors, stacks)
  // The min number of sectors is 3 and the min number of stacks are 2.
  //
  //  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
  // CREATED: 2017-11-01
  // UPDATED: 2020-05-20
  ///////////////////////////////////////////////////////////////////////////////

  clear();
  float x, y, z, xy;
  float nx, ny, nz, length_inv = 1 / radius;
  float s, t;

  const float sector_step = 2 * kPi<float> / num_sectors;
  const float stack_step = kPi<float> / num_stacks;
  float sector_angle, stack_angle;

  for (U i = 0; i <= num_stacks; ++i) {
    stack_angle = kPi<float> / 2 - i * stack_step;
    xy = radius * cos(stack_angle);
    z = radius * sin(stack_angle);

    // add (num_sectors+1) vertices per stack
    // the first and last vertices have same position and normal, but different
    // tex coords
    for (U j = 0; j <= num_sectors; ++j) {
      sector_angle = j * sector_step;

      // vertex position
      x = xy * cos(sector_angle);
      y = xy * sin(sector_angle);
      vertices.push_back(float3{x, y, z});

      // normalized vertex normal
      nx = x * length_inv;
      ny = y * length_inv;
      nz = z * length_inv;
      normals.push_back(float3{nx, ny, nz});

      // vertex tex coord between [0, 1]
      s = static_cast<float>(j) / num_sectors;
      t = static_cast<float>(i) / num_stacks;
      texcoords.push_back(float2{s, t});
    }
  }

  // indices
  //  k1--k1+1
  //  |  / |
  //  | /  |
  //  k2--k2+1
  U k1, k2;
  for (U i = 0; i < num_stacks; ++i) {
    k1 = i * (num_sectors + 1);  // beginning of current stack
    k2 = k1 + num_sectors + 1;   // beginning of next stack

    for (U j = 0; j < num_sectors; ++j, ++k1, ++k2) {
      // 2 triangles per sector excluding 1st and last stacks
      if (i != 0) {
        faces.push_back(U3{k1, k2, k1 + 1});  // k1---k2---k1+1
      }
      if (i != (num_stacks - 1)) {
        faces.push_back(U3{k1 + 1, k2, k2 + 1});  // k1+1---k2---k2+1
      }
    }
  }
}

void Mesh::set_obj(const char* filename) {
  clear();
  std::vector<U3> tex_faces;
  std::vector<U3> normal_faces;
  std::vector<float2> texcoords_compact;
  std::vector<float3> normals_compact;
  std::ifstream file_stream(filename);
  std::stringstream line_stream;
  std::string line;
  std::array<std::string, 4> tokens;
  std::stringstream face_entry_stream;
  std::array<std::string, 3> face_entry_tokens;
  U3 face, tex_face, normal_face;
  int num_tokens;
  int face_token_id;
  file_stream.exceptions(std::ios_base::badbit);
  while (std::getline(file_stream, line)) {
    num_tokens = 0;
    line_stream.clear();
    line_stream.str(line);
    while (num_tokens < 4 &&
           std::getline(line_stream, tokens[num_tokens], ' ')) {
      ++num_tokens;
    }
    if (num_tokens == 0) continue;
    if (tokens[0] == "v") {
      vertices.push_back(from_string<float3>(tokens[1], tokens[2], tokens[3]));
    } else if (tokens[0] == "vt") {
      texcoords_compact.push_back(from_string<float2>(tokens[1], tokens[2]));
    } else if (tokens[0] == "vn") {
      normals_compact.push_back(
          from_string<float3>(tokens[1], tokens[2], tokens[3]));
    } else if (tokens[0] == "f") {
      for (U face_entry_id = 1; face_entry_id <= 3; ++face_entry_id) {
        face_token_id = 0;
        face_entry_stream.clear();
        face_entry_stream.str(tokens[face_entry_id]);
        while (face_token_id < 3 &&
               std::getline(face_entry_stream, face_entry_tokens[face_token_id],
                            '/')) {
          if (face_token_id == 0) {
            if (face_entry_id == 1)
              face.x = from_string<U>(face_entry_tokens[face_token_id]);
            if (face_entry_id == 2)
              face.y = from_string<U>(face_entry_tokens[face_token_id]);
            if (face_entry_id == 3)
              face.z = from_string<U>(face_entry_tokens[face_token_id]);
          } else if (face_token_id == 1) {
            if (face_entry_id == 1)
              tex_face.x = from_string<U>(face_entry_tokens[face_token_id]);
            if (face_entry_id == 2)
              tex_face.y = from_string<U>(face_entry_tokens[face_token_id]);
            if (face_entry_id == 3)
              tex_face.z = from_string<U>(face_entry_tokens[face_token_id]);
          } else if (face_token_id == 2) {
            if (face_entry_id == 1)
              normal_face.x = from_string<U>(face_entry_tokens[face_token_id]);
            if (face_entry_id == 2)
              normal_face.y = from_string<U>(face_entry_tokens[face_token_id]);
            if (face_entry_id == 3)
              normal_face.z = from_string<U>(face_entry_tokens[face_token_id]);
          }
          face_token_id += 1;
        }
      }
      faces.push_back(face - 1);  // OBJ vertex: one-based indexing
      if (face_token_id >= 2) {
        tex_faces.push_back(tex_face - 1);
      }
      if (face_token_id >= 3) {
        normal_faces.push_back(normal_face - 1);
      }
    }
  }
  if (!tex_faces.empty()) {
    texcoords.resize(vertices.size());
    for (U i = 0; i < faces.size(); ++i) {
      U3 const& face = faces[i];
      U3 const& tex_face = tex_faces[i];
      texcoords[face.x] = texcoords_compact[tex_face.x];
      texcoords[face.y] = texcoords_compact[tex_face.y];
      texcoords[face.z] = texcoords_compact[tex_face.z];
    }
  }
  if (!normal_faces.empty()) {
    normals.resize(vertices.size());
    for (U i = 0; i < faces.size(); ++i) {
      U3 const& face = faces[i];
      U3 const& normal_face = normal_faces[i];
      normals[face.x] = normals_compact[normal_face.x];
      normals[face.y] = normals_compact[normal_face.y];
      normals[face.z] = normals_compact[normal_face.z];
    }
  } else {
    calculate_normals();
  }
}
void Mesh::calculate_normals() {
  normals.resize(vertices.size());
  memset(normals.data(), 0, normals.size() * sizeof(float3));
  for (U3 const& face : faces) {
    float3 const& v0 = vertices[face.x];
    float3 const& v1 = vertices[face.y];
    float3 const& v2 = vertices[face.z];
    float3 normal = cross(v1 - v0, v2 - v0);
    normal = normalize(normal);
    normals[face.x] += normal;
    normals[face.y] += normal;
    normals[face.z] += normal;
  }
  for (float3& normal : normals) {
    normal = normalize(normal);
  }
}
void Mesh::clear() {
  vertices.clear();
  normals.clear();
  texcoords.clear();
  faces.clear();
}
}  // namespace alluvion
