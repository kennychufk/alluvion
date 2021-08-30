#ifndef ALLUVION_PILE_HPP
#define ALLUVION_PILE_HPP

#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <vector>

#include "alluvion/dg/cubic_lagrange_discrete_grid.hpp"
#include "alluvion/dg/mesh_distance.hpp"
#include "alluvion/mesh.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

namespace alluvion {
template <typename TF>
class Pile {
 private:
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TRunner = Runner<TF>;
  static dg::MeshDistance<TF3, TF>* construct_mesh_distance(
      VertexList const& vertices, FaceList const& faces) {
    std::vector<dg::Vector3r<TF>> dg_vertices;
    std::vector<std::array<U, 3>> dg_faces;
    dg_vertices.reserve(vertices.size());
    dg_faces.reserve(faces.size());
    for (TF3 const& vertex : vertices) {
      dg_vertices.push_back(dg::Vector3r<TF>(vertex.x, vertex.y, vertex.z));
    }
    for (U3 const& face : faces) {
      dg_faces.push_back({face.x, face.y, face.z});
    }
    return new dg::MeshDistance<TF3, TF>(
        dg::TriangleMesh<TF>(dg_vertices, dg_faces));
  }
  static void calculate_grid_attributes(dg::Distance<TF3, TF> const& distance,
                                        U3 const& resolution, TF margin,
                                        TF3& domain_min, TF3& domain_max,
                                        U& grid_size, TF3& cell_size) {
    domain_min = distance.get_aabb_min() - margin;
    domain_max = distance.get_aabb_max() + margin;
    cell_size = (domain_max - domain_min) / resolution;
    U nv = (resolution.x + 1) * (resolution.y + 1) * (resolution.z + 1);
    U ne_x = (resolution.x + 0) * (resolution.y + 1) * (resolution.z + 1);
    U ne_y = (resolution.x + 1) * (resolution.y + 0) * (resolution.z + 1);
    U ne_z = (resolution.x + 1) * (resolution.y + 1) * (resolution.z + 0);
    U ne = ne_x + ne_y + ne_z;
    grid_size = nv + 2 * ne;
  }
  static std::vector<TF> construct_distance_grid(
      dg::Distance<TF3, TF> const& distance, U3 const& resolution, TF margin,
      TF sign, TF thickness, TF3 const& domain_min, TF3 const& domain_max) {
    dg::CubicLagrangeDiscreteGrid grid_host(
        dg::AlignedBox3r<TF>(
            dg::Vector3r<TF>(domain_min.x, domain_min.y, domain_min.z),
            dg::Vector3r<TF>(domain_max.x, domain_max.y, domain_max.z)),
        {resolution.x, resolution.y, resolution.z});
    grid_host.addFunction([&distance](dg::Vector3r<TF> const& xi) {
      // signedDistanceCached failed for unknown reasons
      return distance.signedDistance(xi);
    });
    std::vector<TF>& nodes = grid_host.node_data()[0];
    dg::Vector3r<TF> dg_cell_size = grid_host.cellSize();
    return nodes;
  }

 public:
  using TContact = Contact<TF3, TF>;
  std::vector<TF> mass_;
  std::vector<TF> restitution_;
  std::vector<TF> friction_;
  std::vector<TF3> inertia_tensor_;

  PinnedVariable<1, TF3> x_;
  std::vector<TF3> oldx_;
  PinnedVariable<1, TF3> v_;
  std::vector<TF3> a_;
  std::vector<TF3> force_;

  PinnedVariable<1, TQ> q_;
  PinnedVariable<1, TF3> omega_;
  std::vector<TF3> torque_;

  std::vector<std::unique_ptr<dg::Distance<TF3, TF>>> distance_list_;
  std::vector<U3> resolution_list_;
  std::vector<TF> sign_list_;
  std::vector<TF> thickness_list_;
  std::vector<std::unique_ptr<Variable<1, TF>>> distance_grids_;
  std::vector<std::unique_ptr<Variable<1, TF>>> volume_grids_;
  std::vector<TF3> domain_min_list_;
  std::vector<TF3> domain_max_list_;
  std::vector<U> grid_size_list_;
  std::vector<TF3> cell_size_list_;

  std::vector<MeshBuffer> mesh_buffer_list_;
  std::vector<std::unique_ptr<Variable<1, TF3>>> collision_vertex_list_;

  std::unique_ptr<Variable<1, TF3>> x_device_;
  std::unique_ptr<Variable<1, TF3>> v_device_;
  std::unique_ptr<Variable<1, TF3>> omega_device_;
  std::unique_ptr<Variable<1, TContact>> contacts_;
  std::unique_ptr<Variable<1, U>> num_contacts_;
  PinnedVariable<1, TContact> contacts_pinned_;
  PinnedVariable<1, U> num_contacts_pinned_;
  TF3 gravity_;

  Store& store_;
  TRunner& runner_;

  Pile(Store& store, TRunner& runner, U max_num_contacts)
      : store_(store),
        runner_(runner),
        x_(store.create_pinned<1, TF3>({0})),
        q_(store.create_pinned<1, TQ>({0})),
        v_(store.create_pinned<1, TF3>({0})),
        omega_(store.create_pinned<1, TF3>({0})),
        x_device_(store.create<1, TF3>({0})),
        v_device_(store.create<1, TF3>({0})),
        omega_device_(store.create<1, TF3>({0})),
        contacts_(store.create<1, TContact>({max_num_contacts})),
        num_contacts_(store.create<1, U>({1})),
        contacts_pinned_(store.create_pinned<1, TContact>({max_num_contacts})),
        num_contacts_pinned_(store.create_pinned<1, U>({1})) {
    store.get_cni().max_num_contacts = max_num_contacts;
  }

  virtual ~Pile() {
    for (U i = 0; i < get_size(); ++i) {
      store_.remove(*distance_grids_[i]);
      store_.remove(*volume_grids_[i]);
      store_.remove(*collision_vertex_list_[i]);
    }
    // PinnedVariable
    store_.remove(x_);
    store_.remove(q_);
    store_.remove(v_);
    store_.remove(omega_);
    store_.remove(contacts_pinned_);
    store_.remove(num_contacts_pinned_);

    store_.remove(*x_device_);
    store_.remove(*v_device_);
    store_.remove(*omega_device_);
    store_.remove(*contacts_);
    store_.remove(*num_contacts_);
  }
  U add(dg::Distance<TF3, TF>* distance, U3 const& resolution, TF sign,
        TF thickness, Mesh const& collision_mesh, TF mass, TF restitution,
        TF friction, TF3 const& inertia_tensor, TF3 const& x, TQ const& q,
        Mesh const& display_mesh) {
    mass_.push_back(mass);
    restitution_.push_back(restitution);
    friction_.push_back(friction);
    inertia_tensor_.push_back(inertia_tensor);

    oldx_.push_back(x);
    a_.push_back(TF3{0, 0, 0});
    force_.push_back(TF3{0, 0, 0});

    torque_.push_back(TF3{0, 0, 0});

    distance_list_.emplace_back(distance);

    resolution_list_.push_back(resolution);
    sign_list_.push_back(sign);
    thickness_list_.push_back(thickness);

    // placeholders
    distance_grids_.emplace_back(store_.create<1, TF>({0}));
    volume_grids_.emplace_back(store_.create<1, TF>({0}));
    domain_min_list_.push_back(TF3{0, 0, 0});
    domain_max_list_.push_back(TF3{0, 0, 0});
    grid_size_list_.push_back(0);
    cell_size_list_.push_back(TF3{0, 0, 0});

    MeshBuffer mesh_buffer;
    if (store_.has_display()) {
      mesh_buffer = store_.get_display()->create_mesh_buffer(display_mesh);
    }
    mesh_buffer_list_.push_back(mesh_buffer);

    Variable<1, TF3>* collision_vertices_var =
        store_.create<1, TF3>({static_cast<U>(collision_mesh.vertices.size())});
    collision_vertex_list_.emplace_back(collision_vertices_var);
    collision_vertices_var->set_bytes(collision_mesh.vertices.data());

    reallocate_kinematics_on_pinned();
    v_(get_size() - 1) = TF3{0, 0, 0};
    x_(get_size() - 1) = x;
    q_(get_size() - 1) = q;
    omega_(get_size() - 1) = TF3{0, 0, 0};
    return get_size() - 1;
  }
  void replace(U i, dg::Distance<TF3, TF>* distance, U3 const& resolution,
               TF sign, TF thickness, Mesh const& collision_mesh, TF mass,
               TF restitution, TF friction, TF3 const& inertia_tensor,
               TF3 const& x, TQ const& q, Mesh const& display_mesh) {
    mass_[i] = mass;
    restitution_[i] = restitution;
    friction_[i] = friction;
    inertia_tensor_[i] = inertia_tensor;

    oldx_[i] = x;
    a_[i] = TF3{0, 0, 0};
    force_[i] = TF3{0, 0, 0};
    torque_[i] = TF3{0, 0, 0};

    distance_list_[i].reset(distance);

    resolution_list_[i] = resolution;
    sign_list_[i] = sign;
    thickness_list_[i] = thickness;

    // placeholders
    store_.remove(*distance_grids_[i]);
    distance_grids_[i].reset(store_.create<1, TF>({0}));
    store_.remove(*volume_grids_[i]);
    volume_grids_[i].reset(store_.create<1, TF>({0}));
    domain_min_list_[i] = TF3{0, 0, 0};
    domain_max_list_[i] = TF3{0, 0, 0};
    grid_size_list_[i] = 0;
    cell_size_list_[i] = TF3{0, 0, 0};

    MeshBuffer mesh_buffer;
    if (store_.has_display()) {
      mesh_buffer = store_.get_display()->create_mesh_buffer(display_mesh);
      store_.get_display()->remove_mesh_buffer(mesh_buffer_list_[i]);
    }
    mesh_buffer_list_[i] = mesh_buffer;

    Variable<1, TF3>* collision_vertices_var =
        store_.create<1, TF3>({static_cast<U>(collision_mesh.vertices.size())});
    store_.remove(*collision_vertex_list_[i]);
    collision_vertex_list_[i].reset(collision_vertices_var);
    collision_vertices_var->set_bytes(collision_mesh.vertices.data());

    v_(get_size() - 1) = TF3{0, 0, 0};
    x_(get_size() - 1) = x;
    q_(get_size() - 1) = q;
    omega_(get_size() - 1) = TF3{0, 0, 0};
  }
  void build_grids(TF margin) {
    store_.copy_cn<TF>();
    for (U i = 0; i < get_size(); ++i) {
      U& num_nodes = grid_size_list_[i];
      TF3& cell_size = cell_size_list_[i];
      TF3& domain_min = domain_min_list_[i];
      TF3& domain_max = domain_max_list_[i];

      calculate_grid_attributes(*distance_list_[i], resolution_list_[i], margin,
                                domain_min, domain_max, num_nodes, cell_size);

      store_.remove(*volume_grids_[i]);
      Variable<1, TF>* volume_grid = store_.create<1, TF>({num_nodes});
      volume_grids_[i].reset(volume_grid);

      volume_grid->set_zero();
      using TBoxDistance = dg::BoxDistance<TF3, TF>;
      using TSphereDistance = dg::SphereDistance<TF3, TF>;
      using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
      if (TBoxDistance const* distance =
              dynamic_cast<TBoxDistance const*>(distance_list_[i].get())) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, 0, sign_list_[i]);
            },
            "update_volume_field(BoxDistance)",
            update_volume_field<TF3, TF, TBoxDistance>);
      } else if (TSphereDistance const* distance =
                     dynamic_cast<TSphereDistance const*>(
                         distance_list_[i].get())) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, 0, sign_list_[i]);
            },
            "update_volume_field(SphereDistance)",
            update_volume_field<TF3, TF, TSphereDistance>);
      } else if (TInfiniteCylinderDistance const* distance =
                     dynamic_cast<TInfiniteCylinderDistance const*>(
                         distance_list_[i].get())) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, 0, sign_list_[i]);
            },
            "update_volume_field(InfiniteCylinderDistance)",
            update_volume_field<TF3, TF, TInfiniteCylinderDistance>);
      } else {
        store_.remove(*distance_grids_[i]);
        Variable<1, TF>* distance_grid = store_.create<1, TF>({num_nodes});
        distance_grids_[i].reset(distance_grid);

        std::vector<TF> nodes_host = construct_distance_grid(
            *distance_list_[i], resolution_list_[i], margin, sign_list_[i],
            thickness_list_[i], domain_min, domain_max);
        distance_grid->set_bytes(nodes_host.data());
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance_grid, domain_min, domain_max,
                  resolution_list_[i], cell_size, num_nodes, 0, sign_list_[i],
                  thickness_list_[i]);
            },
            "update_volume_field", update_volume_field<TF3, TF>);
      }
    }
  }
  void set_gravity(TF3 gravity) { gravity_ = gravity; }
  void reallocate_kinematics_on_device() {
    store_.remove(*x_device_);
    store_.remove(*v_device_);
    store_.remove(*omega_device_);
    x_device_.reset(store_.create<1, TF3>({get_size()}));
    v_device_.reset(store_.create<1, TF3>({get_size()}));
    omega_device_.reset(store_.create<1, TF3>({get_size()}));

    store_.get_cni().num_boundaries = get_size();
  }
  void reallocate_kinematics_on_pinned() {
    PinnedVariable<1, TF3> x_new = store_.create_pinned<1, TF3>({get_size()});
    PinnedVariable<1, TQ> q_new = store_.create_pinned<1, TQ>({get_size()});
    PinnedVariable<1, TF3> v_new = store_.create_pinned<1, TF3>({get_size()});
    PinnedVariable<1, TF3> omega_new =
        store_.create_pinned<1, TF3>({get_size()});
    x_new.set_bytes(x_.ptr_, x_.get_num_bytes());
    q_new.set_bytes(q_.ptr_, q_.get_num_bytes());
    v_new.set_bytes(v_.ptr_, v_.get_num_bytes());
    omega_new.set_bytes(omega_.ptr_, omega_.get_num_bytes());
    store_.remove(x_);
    store_.remove(q_);
    store_.remove(v_);
    store_.remove(omega_);
    x_ = x_new;
    q_ = q_new;
    v_ = v_new;
    omega_ = omega_new;
  }
  void copy_kinematics_to_device() {
    x_device_->set_bytes(x_.ptr_);
    v_device_->set_bytes(v_.ptr_);
    omega_device_->set_bytes(omega_.ptr_);
  }
  void write_file(const char* filename) const {
    std::ofstream stream(filename, std::ios::binary | std::ios::trunc);
    for (U i = 0; i < get_size(); ++i) {
      stream.write(reinterpret_cast<const char*>(&x_(i)), sizeof(TF3));
      stream.write(reinterpret_cast<const char*>(&v_(i)), sizeof(TF3));
      stream.write(reinterpret_cast<const char*>(&q_(i)), sizeof(TQ));
      stream.write(reinterpret_cast<const char*>(&omega_(i)), sizeof(TF3));
    }
  }
  // TODO: rename to load
  void read_file(const char* filename, int num_rigids = -1, U offset = 0) {
    std::ifstream stream(filename, std::ios::binary);
    U num_rigids_to_read = get_size();
    if (num_rigids >= 0) {
      num_rigids_to_read = static_cast<U>(num_rigids);
    }
    num_rigids_to_read = min(get_size(), num_rigids_to_read);
    for (U i = 0; i < num_rigids_to_read; ++i) {
      U offset_id = offset + i;
      stream.read(reinterpret_cast<char*>(&x_(offset_id)), sizeof(TF3));
      stream.read(reinterpret_cast<char*>(&v_(offset_id)), sizeof(TF3));
      stream.read(reinterpret_cast<char*>(&q_(offset_id)), sizeof(TQ));
      stream.read(reinterpret_cast<char*>(&omega_(offset_id)), sizeof(TF3));
      if (stream.peek() == std::ifstream::traits_type::eof()) break;
    }
  }
  static void read(const char* filename, U num_rigids, TF3* x_dst, TF3* v_dst,
                   TQ* q_dst, TF3* omega_dst) {
    std::ifstream stream(filename, std::ios::binary);
    for (U i = 0; i < num_rigids; ++i) {
      stream.read(reinterpret_cast<char*>(x_dst + i), sizeof(TF3));
      stream.read(reinterpret_cast<char*>(v_dst + i), sizeof(TF3));
      stream.read(reinterpret_cast<char*>(q_dst + i), sizeof(TQ));
      stream.read(reinterpret_cast<char*>(omega_dst + i), sizeof(TF3));
      if (stream.peek() == std::ifstream::traits_type::eof()) break;
    }
  }
  void integrate_kinematics(TF dt) {
    for (U i = 0; i < get_size(); ++i) {
      if (mass_[i] == 0) {
        x_(i) += v_(i) * dt;
        q_(i) += dt * calculate_dq(omega_(i), q_(i));
        q_(i) = normalize(q_(i));
      } else {
        v_(i) += 1 / mass_[i] * force_[i] * dt;
        omega_(i) += calculate_angular_acceleration(inertia_tensor_[i], q_(i),
                                                    torque_[i]) *
                     dt;
        a_[i] = gravity_;
        q_(i) += dt * calculate_dq(omega_(i), q_(i));
        q_(i) = normalize(q_(i));
        // x_(i) += (a_[i] * dt + v_(i)) * dt;
        // v_(i) += a_[i] * dt;
        TF3 dx = (a_[i] * dt + v_(i)) * dt;
        x_(i) += dx;
        v_(i) = 1 / dt * dx;
      }
    }
  }
  TF calculate_cfl_v2() const {
    TF max_v2 = std::numeric_limits<TF>::lowest();
    for (U i = 0; i < get_size(); ++i) {
      TF v2 = length_sqr(
          cross(omega_(i), TF3{distance_list_[i]->get_max_distance(), 0, 0}) +
          v_(i));
      if (v2 > max_v2) max_v2 = v2;
    }
    return max_v2;
  }
  void find_contacts(U i, U j) {
    num_contacts_->set_zero();
    Variable<1, TF3>& vertices_i = *collision_vertex_list_[i];
    U num_vertices_i = vertices_i.get_linear_shape();
    runner_.launch_collision_test(
        *distance_list_[j], *distance_grids_[j], i, j, vertices_i,
        *num_contacts_, *contacts_, mass_[i], inertia_tensor_[i], x_(i), v_(i),
        q_(i), omega_(i), mass_[j], inertia_tensor_[j], x_(j), v_(j), q_(j),
        omega_(j), restitution_[i] * restitution_[j],
        friction_[i] + friction_[j], domain_min_list_[j], domain_max_list_[j],
        resolution_list_[j], cell_size_list_[j], 0, sign_list_[j],
        num_vertices_i);
  }
  void find_contacts() {
    num_contacts_->set_zero();
    for (U i = 0; i < get_size(); ++i) {
      Variable<1, TF3>& vertices_i = *collision_vertex_list_[i];
      U num_vertices_i = vertices_i.get_linear_shape();
      for (U j = 0; j < get_size(); ++j) {
        if (i == j || (mass_[i] == 0 and mass_[i] == 0)) continue;
        runner_.launch_collision_test(
            *distance_list_[j], *distance_grids_[j], i, j, vertices_i,
            *num_contacts_, *contacts_, mass_[i], inertia_tensor_[i], x_(i),
            v_(i), q_(i), omega_(i), mass_[j], inertia_tensor_[j], x_(j), v_(j),
            q_(j), omega_(j), restitution_[i] * restitution_[j],
            friction_[i] + friction_[j], domain_min_list_[j],
            domain_max_list_[j], resolution_list_[j], cell_size_list_[j], 0,
            sign_list_[j], num_vertices_i);
      }
    }
  }
  void solve_contacts() {
    num_contacts_->get_bytes(num_contacts_pinned_.ptr_);
    U num_contacts = num_contacts_pinned_(0);
    contacts_->get_bytes(contacts_pinned_.ptr_,
                         num_contacts * sizeof(TContact));
    for (U solve_iter = 0; solve_iter < 5; ++solve_iter) {
      for (U contact_key = 0; contact_key < num_contacts; ++contact_key) {
        TContact& contact = contacts_pinned_(contact_key);
        U i = contact.i;
        U j = contact.j;

        TF mass_i = mass_[i];
        TF mass_j = mass_[j];
        TF3 x_i = x_(i);
        TF3 x_j = x_(j);
        TF3 v_i = v_(i);
        TF3 v_j = v_(j);
        TF3 omega_i = omega_(i);
        TF3 omega_j = omega_(j);

        TF depth =
            dot(contact.n, contact.cp_i - contact.cp_j);  // penetration depth
        TF3 r_i = contact.cp_i - x_i;
        TF3 r_j = contact.cp_j - x_j;

        TF3 u_i = v_i + cross(omega_i, r_i);
        TF3 u_j = v_j + cross(omega_j, r_j);

        TF3 u_rel = u_i - u_j;
        TF u_rel_n = dot(contact.n, u_rel);
        TF delta_u_reln = contact.goalu - u_rel_n;

        TF correction_magnitude = contact.nkninv * delta_u_reln;

        if (correction_magnitude < -contact.impulse_sum) {
          correction_magnitude = -contact.impulse_sum;
        }
        const TF stiffness = 1;
        if (depth < 0) {
          correction_magnitude -= stiffness * contact.nkninv * depth;
        }
        TF3 p = correction_magnitude * contact.n;

        contact.impulse_sum += correction_magnitude;

        // dynamic friction
        TF pn = dot(p, contact.n);
        if (contact.friction * pn > contact.pmax) {
          p -= contact.pmax * contact.t;
        } else if (contact.friction * pn < -contact.pmax) {
          p += contact.pmax * contact.t;
        } else {
          p -= contact.friction * pn * contact.t;
        }

        if (mass_i != 0) {
          v_(i) += p / mass_i;
          omega_(i) += apply_congruent(cross(r_i, p), contact.iiwi_diag,
                                       contact.iiwi_off_diag);
        }
        if (mass_j != 0) {
          v_(j) += -p / mass_j;
          omega_(j) += apply_congruent(cross(r_j, -p), contact.iiwj_diag,
                                       contact.iiwj_off_diag);
        }
      }
    }
  }
  U get_size() const { return distance_grids_.size(); }
  std::vector<Variable<1, TF>> get_distance_grids() const {
    std::vector<Variable<1, TF>> grid_copy;
    grid_copy.reserve(get_size());
    for (std::unique_ptr<Variable<1, TF>> const& grid : distance_grids_) {
      grid_copy.push_back(*grid);
    }
    return grid_copy;
  }
  std::vector<Variable<1, TF>> get_volume_grids() const {
    std::vector<Variable<1, TF>> grid_copy;
    grid_copy.reserve(get_size());
    for (std::unique_ptr<Variable<1, TF>> const& grid : volume_grids_) {
      grid_copy.push_back(*grid);
    }
    return grid_copy;
  }
  glm::mat4 get_matrix(U i) const {
    TQ const& q = q_(i);
    TF3 const& translation = x_(i);
    float column_major_transformation[16] = {
        static_cast<float>(1 - 2 * (q.y * q.y + q.z * q.z)),
        static_cast<float>(2 * (q.x * q.y + q.z * q.w)),
        static_cast<float>(2 * (q.x * q.z - q.y * q.w)),
        static_cast<float>(0),
        static_cast<float>(2 * (q.x * q.y - q.z * q.w)),
        static_cast<float>(1 - 2 * (q.x * q.x + q.z * q.z)),
        static_cast<float>(2 * (q.y * q.z + q.x * q.w)),
        static_cast<float>(0),
        static_cast<float>(2 * (q.x * q.z + q.y * q.w)),
        static_cast<float>(2 * (q.y * q.z - q.x * q.w)),
        static_cast<float>(1 - 2 * (q.x * q.x + q.y * q.y)),
        static_cast<float>(0),
        static_cast<float>(translation.x),
        static_cast<float>(translation.y),
        static_cast<float>(translation.z),
        static_cast<float>(1)};
    return glm::make_mat4(column_major_transformation);
  };

  template <class Lambda>
  void for_each_rigid(Lambda f) {
    for (U i = 0; i < get_size(); ++i) {
      f(i, *distance_list_[i], *distance_grids_[i], *volume_grids_[i], x_(i),
        q_(i), domain_min_list_[i], domain_max_list_[i], resolution_list_[i],
        cell_size_list_[i], grid_size_list_[i], sign_list_[i],
        thickness_list_[i]);
    }
  }
};
}  // namespace alluvion

#endif /* ALLUVION_PILE_HPP */
