#include <glm/gtc/type_ptr.hpp>
#include <vector>

#include "alluvion/mesh.hpp"
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

void Pile::add(Mesh const& field_mesh, U3 const& resolution, F sign,
               F thickness, Mesh const& collision_mesh, F mass, F restitution,
               F friction, F boundary_viscosity, F3 const& inertia_tensor,
               F3 const& x, Q const& q, Mesh const& display_mesh) {
  add(construct_mesh_distance(field_mesh.vertices, field_mesh.faces),
      resolution, sign, thickness, collision_mesh, mass, restitution, friction,
      boundary_viscosity, inertia_tensor, x, q, display_mesh);
}

void Pile::add(dg::Distance* distance, U3 const& resolution, F sign,
               F thickness, Mesh const& collision_mesh, F mass, F restitution,
               F friction, F boundary_viscosity, F3 const& inertia_tensor,
               F3 const& x, Q const& q, Mesh const& display_mesh) {
  mass_.push_back(mass);
  restitution_.push_back(restitution);
  friction_.push_back(friction);
  boundary_viscosity_.push_back(boundary_viscosity);
  inertia_tensor_.push_back(inertia_tensor);

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

  distance_list_.emplace_back(distance);

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

  MeshBuffer mesh_buffer;
  if (store_.has_display()) {
    mesh_buffer = store_.create_mesh_buffer(display_mesh);
  }
  mesh_buffer_list_.push_back(mesh_buffer);

  Variable<1, F3> collision_vertices_var =
      store_.create<1, F3>({static_cast<U>(collision_mesh.vertices.size())});
  collision_vertex_list_.push_back(collision_vertices_var);
  collision_vertices_var.set_bytes(collision_mesh.vertices.data());
}

void Pile::build_grids(F margin) {
  for (U i = 0; i < get_size(); ++i) {
    U& num_nodes = grid_size_list_[i];
    F3& cell_size = cell_size_list_[i];
    F3& domain_min = domain_min_list_[i];
    F3& domain_max = domain_max_list_[i];
    std::vector<F> nodes_host = construct_distance_grid(
        *distance_list_[i], resolution_list_[i], margin, sign_list_[i],
        thickness_list_[i], domain_min, domain_max, num_nodes, cell_size);

    store_.remove(distance_grids_[i]);
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

void Pile::reallocate_kinematics_on_device() {
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

void Pile::copy_kinematics_to_device() {
  x_device_.set_bytes(x_.data());
  v_device_.set_bytes(v_.data());
  omega_device_.set_bytes(omega_.data());
}

U Pile::get_size() const { return distance_grids_.size(); }

glm::mat4 Pile::get_matrix(U i) const {
  Q const& q = q_[i];
  F3 const& translation = x_[i];
  float column_major_transformation[16] = {1 - 2 * (q.y * q.y + q.z * q.z),
                                           2 * (q.x * q.y + q.z * q.w),
                                           2 * (q.x * q.z - q.y * q.w),
                                           0,
                                           2 * (q.x * q.y - q.z * q.w),
                                           1 - 2 * (q.x * q.x + q.z * q.z),
                                           2 * (q.y * q.z + q.x * q.w),
                                           0,
                                           2 * (q.x * q.z + q.y * q.w),
                                           2 * (q.y * q.z - q.x * q.w),
                                           1 - 2 * (q.x * q.x + q.y * q.y),
                                           0,
                                           translation.x,
                                           translation.y,
                                           translation.z,
                                           1};
  return glm::make_mat4(column_major_transformation);
}

dg::MeshDistance* Pile::construct_mesh_distance(VertexList const& vertices,
                                                FaceList const& faces) {
  std::vector<dg::Vector3r> dg_vertices;
  std::vector<std::array<unsigned int, 3>> dg_faces;
  dg_vertices.reserve(vertices.size());
  dg_faces.reserve(faces.size());
  for (F3 const& vertex : vertices) {
    dg_vertices.push_back(dg::Vector3r(vertex.x, vertex.y, vertex.z));
  }
  for (U3 const& face : faces) {
    dg_faces.push_back({face.x, face.y, face.z});
  }
  return new dg::MeshDistance(dg::TriangleMesh(dg_vertices, dg_faces));
}

std::vector<F> Pile::construct_distance_grid(dg::Distance const& distance,
                                             U3 const& resolution, F margin,
                                             F sign, F thickness,
                                             F3& domain_min, F3& domain_max,
                                             U& grid_size, F3& cell_size) {
  domain_min = distance.get_aabb_min() - margin;
  domain_max = distance.get_aabb_max() + margin;
  dg::CubicLagrangeDiscreteGrid grid_host(
      dg::AlignedBox3r(dg::Vector3r(domain_min.x, domain_min.y, domain_min.z),
                       dg::Vector3r(domain_max.x, domain_max.y, domain_max.z)),
      {resolution.x, resolution.y, resolution.z});
  grid_host.addFunction([&distance](dg::Vector3r const& xi) {
    // signedDistanceCached failed for unknown reasons
    return distance.signedDistance(xi);
  });
  std::vector<F>& nodes = grid_host.node_data()[0];
  grid_size = nodes.size();
  dg::Vector3r dg_cell_size = grid_host.cellSize();
  cell_size =
      make_vector<F3>(dg_cell_size(0), dg_cell_size(1), dg_cell_size(2));
  return nodes;
}

}  // namespace alluvion
