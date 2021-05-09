#include <glm/gtc/type_ptr.hpp>
#include <vector>

#include "alluvion/mesh.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"

namespace alluvion {
Pile::Pile(Store& store, U max_num_contacts)
    : store_(store),
      x_(store.create_pinned<1, F3>({0})),
      v_(store.create_pinned<1, F3>({0})),
      omega_(store.create_pinned<1, F3>({0})),
      x_device_(store.create<1, F3>({0})),
      v_device_(store.create<1, F3>({0})),
      omega_device_(store.create<1, F3>({0})),
      boundary_viscosity_device_(store.create<1, F>({0})),
      contacts_(store.create<1, Contact>({max_num_contacts})),
      num_contacts_(store.create<1, U>({1})),
      contacts_pinned_(store.create_pinned<1, Contact>({max_num_contacts})),
      num_contacts_pinned_(store.create_pinned<1, U>({1})) {
  cnst::set_max_num_contacts(max_num_contacts);
}
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
  q_mat_.push_back(q);
  q_initial_.push_back(q);

  oldx_.push_back(x);
  a_.push_back(F3{0, 0, 0});
  force_.push_back(F3{0, 0, 0});

  q_.push_back(q);
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

  reallocate_kinematics_on_pinned();
  v_(get_size() - 1) = F3{0, 0, 0};
  x_(get_size() - 1) = x;
  omega_(get_size() - 1) = F3{0, 0, 0};
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

void Pile::reallocate_kinematics_on_pinned() {
  PinnedVariable<1, F3> x_new = store_.create_pinned<1, F3>({get_size()});
  PinnedVariable<1, F3> v_new = store_.create_pinned<1, F3>({get_size()});
  PinnedVariable<1, F3> omega_new = store_.create_pinned<1, F3>({get_size()});
  x_new.set_bytes(x_.ptr_, x_.get_num_bytes());
  v_new.set_bytes(v_.ptr_, v_.get_num_bytes());
  omega_new.set_bytes(omega_.ptr_, omega_.get_num_bytes());
  store_.remove(x_);
  store_.remove(v_);
  store_.remove(omega_);
  x_ = x_new;
  v_ = v_new;
  omega_ = omega_new;
}

void Pile::copy_kinematics_to_device() {
  x_device_.set_bytes(x_.ptr_);
  v_device_.set_bytes(v_.ptr_);
  omega_device_.set_bytes(omega_.ptr_);
}

void Pile::find_contacts() {
  num_contacts_.set_zero();
  for (U i = 0; i < get_size(); ++i) {
    Variable<1, F3>& vertices_i = collision_vertex_list_[i];
    U num_vertices_i = vertices_i.get_linear_shape();
    for (U j = 0; j < get_size(); ++j) {
      if (i == j || (mass_[i] == 0._F and mass_[i] == 0._F)) continue;
      Runner::launch(num_vertices_i, 256, [&](U grid_size, U block_size) {
        collision_test<<<grid_size, block_size>>>(
            i, j, vertices_i, num_contacts_, contacts_, mass_[i],
            inertia_tensor_[i], x_(i), v_(i), q_[i], omega_(i), mass_[j],
            inertia_tensor_[j], x_(j), v_(j), q_[j], omega_(j), q_initial_[j],
            q_mat_[j], x_mat_[j], restitution_[i] * restitution_[j],
            friction_[i] + friction_[j], distance_grids_[j],
            domain_min_list_[j], domain_max_list_[j], resolution_list_[j],
            cell_size_list_[j], 0, sign_list_[j], num_vertices_i);
      });
    }
  }
}

void Pile::solve_contacts() {
  num_contacts_.get_bytes(num_contacts_pinned_.ptr_);
  U num_contacts = num_contacts_pinned_(0);
  contacts_.get_bytes(contacts_pinned_.ptr_, num_contacts * sizeof(Contact));
  for (U solve_iter = 0; solve_iter < 5; ++solve_iter) {
    for (U contact_key = 0; contact_key < num_contacts; ++contact_key) {
      Contact& contact = contacts_pinned_(contact_key);
      U i = contact.i;
      U j = contact.j;

      F mass_i = mass_[i];
      F mass_j = mass_[j];
      F3 x_i = x_(i);
      F3 x_j = x_(j);
      F3 v_i = v_(i);
      F3 v_j = v_(j);
      F3 omega_i = omega_(i);
      F3 omega_j = omega_(j);

      F depth =
          dot(contact.n, contact.cp_i - contact.cp_j);  // penetration depth
      F3 r_i = contact.cp_i - x_i;
      F3 r_j = contact.cp_j - x_j;

      F3 u_i = v_i + cross(omega_i, r_i);
      F3 u_j = v_j + cross(omega_j, r_j);

      F3 u_rel = u_i - u_j;
      F u_rel_n = dot(contact.n, u_rel);
      F delta_u_reln = contact.goalu - u_rel_n;

      F correction_magnitude = contact.nkninv * delta_u_reln;

      if (correction_magnitude < -contact.impulse_sum) {
        correction_magnitude = -contact.impulse_sum;
      }
      const F stiffness = 1._F;
      if (depth < 0._F) {
        correction_magnitude -= stiffness * contact.nkninv * depth;
      }
      F3 p = correction_magnitude * contact.n;

      contact.impulse_sum += correction_magnitude;

      // dynamic friction
      F pn = dot(p, contact.n);
      if (contact.friction * pn > contact.pmax) {
        p -= contact.pmax * contact.t;
      } else if (contact.friction * pn < -contact.pmax) {
        p += contact.pmax * contact.t;
      } else {
        p -= contact.friction * pn * contact.t;
      }

      if (mass_i != 0._F) {
        v_(i) += p / mass_i;
        omega_(i) += apply_congruent(cross(r_i, p), contact.iiwi_diag,
                                     contact.iiwi_off_diag);
      }
      if (mass_j != 0._F) {
        v_(j) += -p / mass_j;
        omega_(j) += apply_congruent(cross(r_j, -p), contact.iiwj_diag,
                                     contact.iiwj_off_diag);
      }
    }
  }
}

U Pile::get_size() const { return distance_grids_.size(); }

glm::mat4 Pile::get_matrix(U i) const {
  Q const& q = q_[i];
  F3 const& translation = x_(i);
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
