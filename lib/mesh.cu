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
      vertices.push_back(float3a{x, y, z});

      // normalized vertex normal
      nx = x * length_inv;
      ny = y * length_inv;
      nz = z * length_inv;
      normals.push_back(float3a{nx, ny, nz});

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

void Mesh::set_cylinder(float radius, float height, U num_sectors,
                        U num_stacks) {
  ///////////////////////////////////////////////////////////////////////////////
  // Cylinder.cpp
  // ============
  //
  //  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
  // CREATED: 2018-03-27
  // UPDATED: 2020-03-14
  ///////////////////////////////////////////////////////////////////////////////

  // clear memory of prev arrays
  clear();

  float x, y, z;  // vertex position

  // generate 3D vertices of a unit circle on XZ plane
  std::vector<float> unitCircleVertices;
  {
    float sectorStep = 2 * kPi<float> / num_sectors;
    float sectorAngle;  // radian

    for (int i = 0; i <= num_sectors; ++i) {
      sectorAngle = i * sectorStep;
      unitCircleVertices.push_back(cos(sectorAngle));  // x
      unitCircleVertices.push_back(0);                 // y
      unitCircleVertices.push_back(sin(sectorAngle));  // z
    }
  }

  // get normals for cylinder sides
  std::vector<float> sideNormals;
  {
    float sectorStep = 2 * kPi<float> / num_sectors;
    float sectorAngle;  // radian
    // compute the normal vector at 0 degree first
    // rotate (x0,y0,z0) per sector angle
    for (int i = 0; i <= num_sectors; ++i) {
      sectorAngle = i * sectorStep;
      sideNormals.push_back(cos(sectorAngle));  // nx
      sideNormals.push_back(0);                 // ny
      sideNormals.push_back(sin(sectorAngle));  // nz
    }
  }

  // put vertices of side cylinder to array by scaling unit circle
  for (int i = 0; i <= num_stacks; ++i) {
    y = -(height * 0.5f) +
        static_cast<float>(i) / num_stacks * height;      // vertex position y
    float t = 1.0f - static_cast<float>(i) / num_stacks;  // top-to-bottom

    for (int j = 0, k = 0; j <= num_sectors; ++j, k += 3) {
      x = unitCircleVertices[k];
      z = unitCircleVertices[k + 2];
      vertices.push_back(float3a{x * radius, y, z * radius});  // position
      normals.push_back(float3a{sideNormals[k], sideNormals[k + 1],
                                sideNormals[k + 2]});  // normal
      texcoords.push_back(
          float2{static_cast<float>(j) / num_sectors, t});  // tex coord
    }
  }

  // remember where the base.top vertices start
  U baseVertexIndex = static_cast<U>(vertices.size());

  // put vertices of base of cylinder
  y = -height * 0.5f;
  vertices.push_back(float3a{0, y, 0});
  normals.push_back(float3a{0, -1, 0});
  texcoords.push_back(float2{0.5f, 0.5f});
  for (int i = 0, j = 0; i < num_sectors; ++i, j += 3) {
    x = unitCircleVertices[j];
    z = unitCircleVertices[j + 2];
    vertices.push_back(float3a{x * radius, y, z * radius});
    normals.push_back(float3a{0, -1, 0});
    texcoords.push_back(
        float2{-x * 0.5f + 0.5f, -z * 0.5f + 0.5f});  // flip horizontal
  }

  // remember where the base vertices start
  U topVertexIndex = static_cast<U>(vertices.size());

  // put vertices of top of cylinder
  y = height * 0.5f;
  vertices.push_back(float3a{0, y, 0});
  normals.push_back(float3a{0, 1, 0});
  texcoords.push_back(float2{0.5f, 0.5f});
  for (int i = 0, j = 0; i < num_sectors; ++i, j += 3) {
    x = unitCircleVertices[j];
    z = unitCircleVertices[j + 2];
    vertices.push_back(float3a{x * radius, y, z * radius});
    normals.push_back(float3a{0, 1, 0});
    texcoords.push_back(float2{x * 0.5f + 0.5f, -z * 0.5f + 0.5f});
  }

  // put indices for sides
  U k1, k2;
  for (U i = 0; i < num_stacks; ++i) {
    k1 = i * (num_sectors + 1);  // bebinning of current stack
    k2 = k1 + num_sectors + 1;   // beginning of next stack

    for (U j = 0; j < num_sectors; ++j, ++k1, ++k2) {
      // 2 trianles per sector
      faces.push_back(U3{k1, k2, k1 + 1});
      faces.push_back(U3{k2, k2 + 1, k1 + 1});
    }
  }

  for (U i = 0, k = baseVertexIndex + 1; i < num_sectors; ++i, ++k) {
    if (i < (num_sectors - 1))
      faces.push_back(U3{baseVertexIndex, k, k + 1});
    else  // last triangle
      faces.push_back(U3{baseVertexIndex, k, baseVertexIndex + 1});
  }

  for (U i = 0, k = topVertexIndex + 1; i < num_sectors; ++i, ++k) {
    if (i < (num_sectors - 1))
      faces.push_back(U3{topVertexIndex, k + 1, k});
    else
      faces.push_back(U3{topVertexIndex, topVertexIndex + 1, k});
  }
}

void Mesh::set_box(float3a widths, U n) {
  // https://github.com/glumpy/glumpy/blob/master/glumpy/geometry/primitives.py
  //  -----------------------------------------------------------------------------
  //  Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
  //  Distributed under the (new) BSD License.
  //  -----------------------------------------------------------------------------
  clear();

  float interval = 1.0f / (n - 1);
  for (U i = 0; i < 6; ++i) {
    for (U j = 0; j < n; ++j) {
      float y = -0.5f + interval * j;
      for (U k = 0; k < n; ++k) {
        float x = -0.5f + interval * k;
        float3a normalized_coord;
        if (i == 0) {
          normalized_coord = float3a{x, y, 0.5};
          normals.push_back(float3a{0, 0, 1});
        } else if (i == 1) {
          normalized_coord = float3a{x, y, -0.5};
          normals.push_back(float3a{0, 0, -1});
        } else if (i == 2) {
          normalized_coord = float3a{0.5, x, y};
          normals.push_back(float3a{1, 0, 0});
        } else if (i == 3) {
          normalized_coord = float3a{-0.5, x, y};
          normals.push_back(float3a{-1, 0, 0});
        } else if (i == 4) {  // y face switched
          normalized_coord = float3a{x, -0.5, y};
          normals.push_back(float3a{0, -1, 0});
        } else if (i == 5) {  // y face switched
          normalized_coord = float3a{x, 0.5, y};
          normals.push_back(float3a{0, 1, 0});
        }
        vertices.push_back(normalized_coord * widths);
        texcoords.push_back(float2{interval * k, interval * j});
      }
    }
  }

  std::vector<U> rectangular_index_tail_removed;
  rectangular_index_tail_removed.reserve((n - 1) * (n - 1));
  for (U i = 0; i < (n * (n - 1)); ++i) {
    if ((i + 1) % n != 0) {
      rectangular_index_tail_removed.push_back(i);
    }
  }

  std::vector<U> repeated_index6;
  for (U i = 0; i < (n - 1) * (n - 1) * 6; ++i) {
    U inner = i % 6;
    U mid = (i / 6) % (n - 1);
    U outer = i / (6 * (n - 1));
    U offset = 0;
    if (inner == 1) offset = 1;
    if (inner == 2 || inner == 4) offset = n + 1;
    if (inner == 5) offset = n;
    repeated_index6.push_back(
        rectangular_index_tail_removed[mid * (n - 1) + outer] + offset);
  }
  for (U i = 0; i < repeated_index6.size() * 6; i += 3) {
    U inner = i % repeated_index6.size();
    U outer = i / repeated_index6.size();
    U offset = n * n * outer;
    bool positive_face = (outer % 2 == 0);  // invert for positive face
    faces.push_back(
        U3{repeated_index6[inner] + offset,
           repeated_index6[inner + (positive_face ? 1 : 2)] + offset,
           repeated_index6[inner + (positive_face ? 2 : 1)] + offset});
  }
}

void Mesh::translate(float3a dx) {
  for (float3a& vertex : vertices) {
    vertex += dx;
  }
}

void Mesh::set_obj(const char* filename) {
  clear();
  std::vector<U3> tex_faces;
  std::vector<U3> normal_faces;
  std::vector<float2> texcoords_compact;
  std::vector<float3a> normals_compact;
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
      vertices.push_back(from_string<float3a>(tokens[1], tokens[2], tokens[3]));
    } else if (tokens[0] == "vt") {
      texcoords_compact.push_back(from_string<float2>(tokens[1], tokens[2]));
    } else if (tokens[0] == "vn") {
      normals_compact.push_back(
          from_string<float3a>(tokens[1], tokens[2], tokens[3]));
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
void Mesh::export_obj(const char* filename) {
  std::ofstream stream(filename, std::ios::trunc);
  stream.precision(std::numeric_limits<float>::max_digits10);
  for (float3a const& vertex : vertices) {
    stream << "v " << vertex.x << " " << vertex.y << " " << vertex.z
           << std::endl;
  }
  for (U3 const& face : faces) {
    stream << "f " << (face.x + 1) << " " << (face.y + 1) << " " << (face.z + 1)
           << std::endl;
  }
}
void Mesh::calculate_normals() {
  normals.resize(vertices.size());
  memset(normals.data(), 0, normals.size() * sizeof(float3a));
  for (U3 const& face : faces) {
    float3a const& v0 = vertices[face.x];
    float3a const& v1 = vertices[face.y];
    float3a const& v2 = vertices[face.z];
    float3a normal = cross(v1 - v0, v2 - v0);
    normal = normalize(normal);
    normals[face.x] += normal;
    normals[face.y] += normal;
    normals[face.z] += normal;
  }
  for (float3a& normal : normals) {
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
