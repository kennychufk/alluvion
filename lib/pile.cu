#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"

namespace alluvion {
Pile::Pile(Store& store)
    : store_(store),
      x_device_(store.create<1, F3>({0})),
      v_device_(store.create<1, F3>({0})),
      omega_device_(store.create<1, F3>({0})),
      boundary_viscosity_device_(store.create<1, F>({0})) {}
Pile::~Pile() {}
void Pile::add(VertexList const& field_vertices, FaceList const& field_faces,
               U3 const& resolution, F sign, F thickness,
               VertexList const& collision_vertices, F mass, F restitution,
               F friction, F boundary_viscosity, F3 const& inertia_tensor,
               F3 const& x, Q const& q, VertexList const& display_vertices,
               FaceList const& display_faces) {
  mass_.push_back(mass);
  restitution_.push_back(restitution);
  friction_.push_back(friction);
  boundary_viscosity_.push_back(boundary_viscosity);
  inertia_tensor_.push_back(inertia_tensor);

  max_dist_.push_back(find_max_distance(field_vertices));
  x_mat_.push_back(F3{0, 0, 0});
  q_mat_.push_back(Q{0, 0, 0, 1});
  q_initial_.push_back(Q{0, 0, 0, 1});

  x_.push_back(x);
  oldx_.push_back(x);
  v_.push_back(F3{0, 0, 0});
  a_.push_back(F3{0, 0, 0});
  force_.push_back(F3{0, 0, 0});

  q_.push_back(q);
  omega_.push_back(F3{0, 0, 0});
  torque_.push_back(F3{0, 0, 0});

  F3 aabb_min;
  F3 aabb_max;
  mesh_distance_list_.push_back(
      construct_mesh_distance(field_vertices, field_faces, aabb_min, aabb_max));
  aabb_min_list_.push_back(aabb_min);
  aabb_max_list_.push_back(aabb_max);
  std::cout << "aabb_min = " << aabb_min.x << " " << aabb_min.y << " "
            << aabb_min.z << std::endl;
  std::cout << "aabb_max = " << aabb_max.x << " " << aabb_max.y << " "
            << aabb_max.z << std::endl;

  resolution_list_.push_back(resolution);
  sign_list_.push_back(sign);
  thickness_list_.push_back(thickness);

  // placeholders
  distance_grids_.push_back(store_.create<1, F>({0}));
  volume_grids_.push_back(store_.create<1, F>({0}));
  domain_min_list_.push_back(F3{0, 0, 0});
  domain_max_list_.push_back(F3{0, 0, 0});
  grid_size_list_.push_back(0);
  cell_size_list_.push_back(F3{0, 0, 0});

  display_vertex_lists_.push_back(display_vertices);
  display_face_lists_.push_back(display_faces);
  Variable<1, F3> collision_vertices_var =
      store_.create<1, F3>({static_cast<U>(collision_vertices.size())});
  collision_vertex_lists_.push_back(collision_vertices_var);
  collision_vertices_var.set_bytes(collision_vertices.data());

  store_.remove(x_device_);
  store_.remove(v_device_);
  store_.remove(omega_device_);
  store_.remove(boundary_viscosity_device_);
  x_device_ = store_.create<1, F3>({get_size()});
  v_device_ = store_.create<1, F3>({get_size()});
  omega_device_ = store_.create<1, F3>({get_size()});
  boundary_viscosity_device_ = store_.create<1, F>({get_size()});

  boundary_viscosity_device_.set_bytes(boundary_viscosity_.data());
}

void Pile::add(const char* field_mesh_filename, U3 const& resolution, F sign,
               F thickness, const char* collision_mesh_filename, F mass,
               F restitution, F friction, F boundary_viscosity,
               F3 const& inertia_tensor, F3 const& x, Q const& q,
               const char* display_mesh_filename) {
  VertexList field_vertices, collision_vertices, display_vertices;
  FaceList field_faces, display_faces;
  read_obj(field_mesh_filename, &field_vertices, &field_faces);
  if (collision_mesh_filename)
    read_obj(collision_mesh_filename, &collision_vertices, nullptr);
  if (display_mesh_filename)
    read_obj(display_mesh_filename, &display_vertices, &display_faces);
  add(field_vertices, field_faces, resolution, sign, thickness,
      collision_vertices, mass, restitution, friction, boundary_viscosity,
      inertia_tensor, x, q, display_vertices, display_faces);
}

U Pile::get_size() const { return distance_grids_.size(); }

void Pile::build_grids(F margin) {
  for (U i = 0; i < get_size(); ++i) {
    U& num_nodes = grid_size_list_[i];
    F3& cell_size = cell_size_list_[i];
    F3& domain_min = domain_min_list_[i];
    F3& domain_max = domain_max_list_[i];
    std::vector<F> nodes_host = construct_distance_grid(
        mesh_distance_list_[i], resolution_list_[i], aabb_min_list_[i],
        aabb_max_list_[i], margin, sign_list_[i], thickness_list_[i],
        domain_min, domain_max, num_nodes, cell_size);

    std::cout << "removing distance grids" << std::endl;
    store_.remove(distance_grids_[i]);
    std::cout << "removed distance grids" << std::endl;
    store_.remove(volume_grids_[i]);
    Variable<1, F> distance_grid = store_.create<1, F>({num_nodes});
    Variable<1, F> volume_grid = store_.create<1, F>({num_nodes});
    distance_grids_[i] = distance_grid;
    volume_grids_[i] = volume_grid;

    distance_grid.set_bytes(nodes_host.data());
    volume_grid.set_zero();
    Runner::launch(num_nodes, 256, [&](U grid_size, U block_size) {
      update_volume_field<<<grid_size, block_size>>>(
          volume_grid, distance_grid, domain_min, domain_max,
          resolution_list_[i], cell_size, num_nodes, 0, sign_list_[i],
          thickness_list_[i]);
    });
  }
}

void Pile::copy_kinematics_to_device() {
  x_device_.set_bytes(x_.data());
  v_device_.set_bytes(v_.data());
  omega_device_.set_bytes(omega_.data());
}

F Pile::find_max_distance(VertexList const& vertices) {
  F max_distance2 = 0;
  for (F3 const& vertex : vertices) {
    F distance2 = length_sqr(vertex);
    if (distance2 > max_distance2) {
      max_distance2 = distance2;
    }
  }
  return sqrt(max_distance2);
}

dg::MeshDistance Pile::construct_mesh_distance(VertexList const& vertices,
                                               FaceList const& faces,
                                               F3& aabb_min, F3& aabb_max) {
  std::vector<dg::Vector3r> dg_vertices;
  std::vector<std::array<unsigned int, 3>> dg_faces;
  dg_vertices.reserve(vertices.size());
  dg_faces.reserve(faces.size());
  aabb_min.x = aabb_min.y = aabb_min.z = std::numeric_limits<F>::max();
  aabb_max.x = aabb_max.y = aabb_max.z = std::numeric_limits<F>::lowest();
  for (F3 const& vertex : vertices) {
    dg_vertices.push_back(dg::Vector3r(vertex.x, vertex.y, vertex.z));
    if (vertex.x < aabb_min.x) aabb_min.x = vertex.x;
    if (vertex.y < aabb_min.y) aabb_min.y = vertex.y;
    if (vertex.z < aabb_min.z) aabb_min.z = vertex.z;
    if (vertex.x > aabb_max.x) aabb_max.x = vertex.x;
    if (vertex.y > aabb_max.y) aabb_max.y = vertex.y;
    if (vertex.z > aabb_max.z) aabb_max.z = vertex.z;
  }
  for (U3 const& face : faces) {
    dg_faces.push_back({face.x, face.y, face.z});
  }
  return dg::MeshDistance(dg::TriangleMesh(dg_vertices, dg_faces));
}

std::vector<F> Pile::construct_distance_grid(
    dg::MeshDistance const& mesh_distance, U3 const& resolution,
    F3 const& aabb_min, F3 const& aabb_max, F margin, F sign, F thickness,
    F3& domain_min, F3& domain_max, U& grid_size, F3& cell_size) {
  domain_min = aabb_min - margin;
  domain_max = aabb_max + margin;
  std::cout << "constructing grid_host" << std::endl;
  dg::CubicLagrangeDiscreteGrid grid_host(
      dg::AlignedBox3r(dg::Vector3r(domain_min.x, domain_min.y, domain_min.z),
                       dg::Vector3r(domain_max.x, domain_max.y, domain_max.z)),
      {resolution.x, resolution.y, resolution.z});
  std::cout << "constructed grid_host" << std::endl;
  grid_host.addFunction([&mesh_distance](dg::Vector3r const& xi) {
    // signedDistanceCached failed for unknown reasons
    return mesh_distance.signedDistance(xi);
  });
  std::cout << "added mesh distance grid" << std::endl;
  std::vector<F>& nodes = grid_host.node_data()[0];
  grid_size = nodes.size();
  dg::Vector3r dg_cell_size = grid_host.cellSize();
  cell_size =
      make_vector<F3>(dg_cell_size(0), dg_cell_size(1), dg_cell_size(2));
  return nodes;
}

void Pile::read_obj(const char* filename, VertexList* vertices,
                    FaceList* faces) {
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
    if (tokens[0] == "v" && vertices) {
      vertices->push_back(from_string<F3>(tokens[1], tokens[2], tokens[3]));
    } else if (tokens[0] == "f" && faces) {
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
      faces->push_back(face - 1);  // OBJ vertex: one-based indexing
    }
  }
}
}  // namespace alluvion
