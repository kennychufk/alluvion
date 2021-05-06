#include <array>
#include <fstream>
#include <sstream>
#include <string>

#include "alluvion/constants.hpp"
#include "alluvion/mesh.hpp"
#include "alluvion/runner.hpp"
namespace alluvion {
Mesh::Mesh() {}
void Mesh::set_uv_sphere(F radius, U num_sectors, U num_stacks) {
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
  F x, y, z, xy;
  F nx, ny, nz, length_inv = 1 / radius;
  F s, t;

  const F sector_step = 2 * kPi<F> / num_sectors;
  const F stack_step = kPi<F> / num_stacks;
  F sector_angle, stack_angle;

  for (U i = 0; i <= num_stacks; ++i) {
    stack_angle = kPi<F> / 2 - i * stack_step;
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
      vertices.push_back(F3{x, y, z});

      // normalized vertex normal
      nx = x * length_inv;
      ny = y * length_inv;
      nz = z * length_inv;
      normals.push_back(F3{nx, ny, nz});

      // vertex tex coord between [0, 1]
      s = static_cast<F>(j) / num_sectors;
      t = static_cast<F>(i) / num_stacks;
      texcoords.push_back(F2{s, t});
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
      vertices.push_back(from_string<F3>(tokens[1], tokens[2], tokens[3]));
    } else if (tokens[0] == "f") {
      for (U face_entry_id = 1; face_entry_id <= 3; ++face_entry_id) {
        face_token_id = 0;
        face_entry_stream.clear();
        face_entry_stream.str(tokens[face_entry_id]);
        while (
            face_token_id < 3 &&
            std::getline(face_entry_stream, face_entry_tokens[face_token_id])) {
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
    }
  }
}
void Mesh::clear() {
  vertices.clear();
  normals.clear();
  texcoords.clear();
  faces.clear();
}
}  // namespace alluvion
