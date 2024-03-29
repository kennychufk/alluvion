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
enum class VolumeMethod { volume_map, pellets };

template <typename TF>
// TODO: different Pile classes for volume_map & pellets
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
  static void calculate_domain_and_resolution(
      dg::Distance<TF3, TF> const& distance, TF margin, TF cell_width, TF sign,
      Const<TF> const& cn, TF3& domain_min, TF3& domain_max, U3& resolution) {
    if (std::isinf(margin)) {
      margin = sign < 0 ? cn.kernel_radius : cn.kernel_radius * 2;
    }
    domain_min = distance.get_aabb_min() - margin;
    domain_max = distance.get_aabb_max() + margin;

    // automatic derivation of resolution in case of 0
    if (resolution.x == 0 || resolution.y == 0 || resolution.z == 0) {
      TF3 domain_extent = domain_max - domain_min;
      if (cell_width == 0) {
        cell_width = cn.particle_radius;
      }
      resolution.x = (resolution.x == 0)
                         ? static_cast<U>(domain_extent.x / cell_width)
                         : resolution.x;
      resolution.y = (resolution.y == 0)
                         ? static_cast<U>(domain_extent.y / cell_width)
                         : resolution.y;
      resolution.z = (resolution.z == 0)
                         ? static_cast<U>(domain_extent.z / cell_width)
                         : resolution.z;
    }
  }
  static void calculate_grid_attributes(dg::Distance<TF3, TF> const& distance,
                                        U3 const& resolution,
                                        TF3 const& domain_min,
                                        TF3 const& domain_max, U& grid_size,
                                        TF3& cell_size) {
    cell_size = (domain_max - domain_min) / resolution;
    U nv = (resolution.x + 1) * (resolution.y + 1) * (resolution.z + 1);
    U ne_x = (resolution.x + 0) * (resolution.y + 1) * (resolution.z + 1);
    U ne_y = (resolution.x + 1) * (resolution.y + 0) * (resolution.z + 1);
    U ne_z = (resolution.x + 1) * (resolution.y + 1) * (resolution.z + 0);
    U ne = ne_x + ne_y + ne_z;
    grid_size = nv + 2 * ne;
  }
  static std::vector<TF> construct_distance_grid(
      dg::Distance<TF3, TF> const& distance, U3 const& resolution, TF sign,
      TF3 const& domain_min, TF3 const& domain_max) {
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
  PinnedVariable<1, TF3> v_;
  std::vector<TF3> a_;
  std::vector<TF3> force_;

  PinnedVariable<1, TQ> q_;
  PinnedVariable<1, TF3> omega_;
  std::vector<TF3> torque_;

  const VolumeMethod volume_method_;
  std::vector<std::unique_ptr<dg::Distance<TF3, TF>>> distance_list_;
  std::vector<U3> resolution_list_;
  std::vector<TF> sign_list_;
  std::vector<std::unique_ptr<Variable<1, TF>>> distance_grids_;
  std::vector<std::unique_ptr<Variable<1, TF>>> volume_grids_;
  std::vector<TF3> domain_min_list_;
  std::vector<TF3> domain_max_list_;
  std::vector<U> grid_size_list_;
  std::vector<TF3> cell_size_list_;
  std::vector<U2> identical_sequence_list_;

  const U max_num_pellets_;
  std::vector<MeshBuffer> mesh_buffer_list_;
  std::vector<std::unique_ptr<Variable<1, TF3>>> collision_vertex_list_;
  std::unique_ptr<Variable<1, U>> pellet_id_to_rigid_id_;
  std::vector<U> pellet_index_offset_list_;
  U num_pellets_;
  U num_collision_pellets_;

  std::unique_ptr<Variable<1, TF3>> x_device_;
  std::unique_ptr<Variable<1, TF3>> v_device_;
  std::unique_ptr<Variable<1, TQ>> q_device_;
  std::unique_ptr<Variable<1, TF3>> omega_device_;
  std::unique_ptr<Variable<1, TF3>> mass_contact_device_;
  std::unique_ptr<Variable<1, TF3>> inertia_device_;
  std::unique_ptr<Variable<1, TContact>> contacts_;
  std::unique_ptr<Variable<1, U>> num_contacts_;
  PinnedVariable<1, TContact> contacts_pinned_;
  PinnedVariable<1, U> num_contacts_pinned_;
  TF3 gravity_;

  Store& store_;
  TRunner& runner_;
  // TODO: use shared pointer
  Const<TF> const* cn_;
  ConstiN* cni_;

  // NOTE: require defined kernel_radius in store's host-side Const<TF>
  Pile(Store& store, TRunner& runner, U max_num_contacts,
       VolumeMethod volume_method = VolumeMethod::volume_map,
       U max_num_pellets_arg = 10000, Const<TF>* cn = nullptr,
       ConstiN* cni = nullptr)
      : store_(store),
        runner_(runner),
        x_(store.create_pinned<1, TF3>({0})),
        q_(store.create_pinned<1, TQ>({0})),
        v_(store.create_pinned<1, TF3>({0})),
        omega_(store.create_pinned<1, TF3>({0})),
        volume_method_(volume_method),
        max_num_pellets_(
            volume_method == VolumeMethod::pellets ? max_num_pellets_arg : 0),
        num_pellets_(0),
        num_collision_pellets_(0),
        pellet_id_to_rigid_id_(store.create<1, U>({0})),
        x_device_(store.create<1, TF3>({0})),
        v_device_(store.create<1, TF3>({0})),
        q_device_(store.create<1, TQ>({0})),
        omega_device_(store.create<1, TF3>({0})),
        mass_contact_device_(store.create<1, TF3>({0})),
        inertia_device_(store.create<1, TF3>({0})),
        contacts_(store.create<1, TContact>({max_num_contacts})),
        num_contacts_(store.create<1, U>({1})),
        contacts_pinned_(store.create_pinned<1, TContact>({max_num_contacts})),
        num_contacts_pinned_(store.create_pinned<1, U>({1})),
        cn_(cn == nullptr ? &store.get_cn<TF>() : cn),
        cni_(cni == nullptr ? &store.get_cni() : cni) {
    if (cn == nullptr) {
      cn = &(store.get_cn<TF>());
    }
    if (cni == nullptr) {
      cni = &(store.get_cni());
    }
    cni->max_num_contacts = max_num_contacts;
    cn->set_cubic_discretization_constants();
    if (cn == nullptr) {
      store.copy_cn<TF>();
    } else {
      store.copy_cn_external(*cn, *cni);
    }
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
    store_.remove(*pellet_id_to_rigid_id_);

    store_.remove(*x_device_);
    store_.remove(*v_device_);
    store_.remove(*q_device_);
    store_.remove(*omega_device_);
    store_.remove(*contacts_);
    store_.remove(*num_contacts_);
  }

  U add(dg::Distance<TF3, TF>* distance, U3 resolution = U3{0},
        TF cell_width = 0, TF margin = std::numeric_limits<TF>::infinity(),
        TF sign = 1, Variable<1, TF3> const& pellets = Variable<1, TF3>(),
        TF mass = 0, TF restitution = 1, TF friction = 0,
        TF3 const& inertia_tensor = TF3{1, 1, 1}, TF3 const& x = TF3{0},
        TQ const& q = TQ{0, 0, 0, 1}, Mesh const& display_mesh = Mesh()) {
    mass_.push_back(mass);
    restitution_.push_back(restitution);
    friction_.push_back(friction);
    inertia_tensor_.push_back(inertia_tensor);

    a_.push_back(TF3{0, 0, 0});
    force_.push_back(TF3{0, 0, 0});
    torque_.push_back(TF3{0, 0, 0});

    TF3 domain_min, domain_max;
    calculate_domain_and_resolution(*distance, margin, cell_width, sign, *cn_,
                                    domain_min, domain_max, resolution);
    distance_list_.emplace_back(distance);
    resolution_list_.push_back(resolution);
    sign_list_.push_back(sign);
    domain_min_list_.push_back(domain_min);
    domain_max_list_.push_back(domain_max);

    // placeholders
    distance_grids_.emplace_back(store_.create<1, TF>({0}));
    volume_grids_.emplace_back(store_.create<1, TF>({0}));
    grid_size_list_.push_back(0);
    cell_size_list_.push_back(TF3{0, 0, 0});

    reallocate_kinematics_on_pinned();
    U boundary_id = get_size() - 1;
    v_(boundary_id) = TF3{0, 0, 0};
    x_(boundary_id) = x;
    q_(boundary_id) = q;
    omega_(boundary_id) = TF3{0, 0, 0};

    MeshBuffer mesh_buffer;
    if (store_.has_display()) {
      mesh_buffer = store_.get_display()->create_mesh_buffer(display_mesh);
    }
    mesh_buffer_list_.push_back(mesh_buffer);

    Variable<1, TF3>* collision_vertices_var =
        store_.create<1, TF3>({pellets.get_linear_shape()});
    collision_vertex_list_.emplace_back(collision_vertices_var);
    collision_vertices_var->set_from(pellets);
    // NOTE: particle_x is always populated with pellets. The decision whether
    // to loop over pellets is controlled by num_pellets
    if (volume_method_ == VolumeMethod::pellets) {
      num_pellets_ += collision_vertices_var->get_linear_shape();
      if (num_pellets_ > max_num_pellets_) {
        std::stringstream error_sstream;
        error_sstream << "Cannot add pellets because the new pellet count "
                      << num_pellets_ << " exceeds the limit "
                      << max_num_pellets_;
        throw std::runtime_error(error_sstream.str());
      }
    }
    num_collision_pellets_ += collision_vertices_var->get_linear_shape();
    store_.remove(*pellet_id_to_rigid_id_);
    pellet_id_to_rigid_id_.reset(store_.create<1, U>({num_collision_pellets_}));
    pellet_index_offset_list_.resize(get_size());
    update_pellet_id_to_rigid_id();

    build_grid(boundary_id);
    return boundary_id;
  }
  U add(dg::Distance<TF3, TF>* distance, U3 resolution = U3{0},
        TF cell_width = 0, TF margin = std::numeric_limits<TF>::infinity(),
        TF sign = 1, Mesh const& collision_mesh = Mesh(), TF mass = 0,
        TF restitution = 1, TF friction = 0,
        TF3 const& inertia_tensor = TF3{1, 1, 1}, TF3 const& x = TF3{0},
        TQ const& q = TQ{0, 0, 0, 1}, Mesh const& display_mesh = Mesh()) {
    mass_.push_back(mass);
    restitution_.push_back(restitution);
    friction_.push_back(friction);
    inertia_tensor_.push_back(inertia_tensor);

    a_.push_back(TF3{0, 0, 0});
    force_.push_back(TF3{0, 0, 0});
    torque_.push_back(TF3{0, 0, 0});

    TF3 domain_min, domain_max;
    calculate_domain_and_resolution(*distance, margin, cell_width, sign, *cn_,
                                    domain_min, domain_max, resolution);
    distance_list_.emplace_back(distance);
    resolution_list_.push_back(resolution);
    sign_list_.push_back(sign);
    domain_min_list_.push_back(domain_min);
    domain_max_list_.push_back(domain_max);

    // placeholders
    distance_grids_.emplace_back(store_.create<1, TF>({0}));
    volume_grids_.emplace_back(store_.create<1, TF>({0}));
    grid_size_list_.push_back(0);
    cell_size_list_.push_back(TF3{0, 0, 0});

    reallocate_kinematics_on_pinned();
    U boundary_id = get_size() - 1;
    v_(boundary_id) = TF3{0, 0, 0};
    x_(boundary_id) = x;
    q_(boundary_id) = q;
    omega_(boundary_id) = TF3{0, 0, 0};

    MeshBuffer mesh_buffer;
    if (store_.has_display()) {
      mesh_buffer = store_.get_display()->create_mesh_buffer(display_mesh);
    }
    mesh_buffer_list_.push_back(mesh_buffer);

    Variable<1, TF3>* collision_vertices_var =
        store_.create<1, TF3>({static_cast<U>(collision_mesh.vertices.size())});
    collision_vertex_list_.emplace_back(collision_vertices_var);
    cast_vertices(*collision_vertices_var, collision_mesh.vertices);
    // NOTE: particle_x is always populated with pellets. The decision whether
    // to loop over pellets is controlled by num_pellets
    if (volume_method_ == VolumeMethod::pellets) {
      num_pellets_ += collision_vertices_var->get_linear_shape();
      if (num_pellets_ > max_num_pellets_) {
        std::stringstream error_sstream;
        error_sstream << "Cannot add pellets because the new pellet count "
                      << num_pellets_ << " exceeds the limit "
                      << max_num_pellets_;
        throw std::runtime_error(error_sstream.str());
      }
    }
    num_collision_pellets_ += collision_vertices_var->get_linear_shape();
    store_.remove(*pellet_id_to_rigid_id_);
    pellet_id_to_rigid_id_.reset(store_.create<1, U>({num_collision_pellets_}));
    pellet_index_offset_list_.resize(get_size());
    update_pellet_id_to_rigid_id();

    build_grid(boundary_id);
    return boundary_id;
  }
  void replace(U i, dg::Distance<TF3, TF>* distance, U3 resolution = U3{0},
               TF cell_width = 0,
               TF margin = std::numeric_limits<TF>::infinity(), TF sign = 1,
               Variable<1, TF3> const& pellets = Variable<1, TF3>(),
               TF mass = 0, TF restitution = 1, TF friction = 0,
               TF3 const& inertia_tensor = TF3{1, 1, 1}, TF3 const& x = TF3{0},
               TQ const& q = TQ{0, 0, 0, 1},
               Mesh const& display_mesh = Mesh()) {
    mass_[i] = mass;
    restitution_[i] = restitution;
    friction_[i] = friction;
    inertia_tensor_[i] = inertia_tensor;

    a_[i] = TF3{0, 0, 0};
    force_[i] = TF3{0, 0, 0};
    torque_[i] = TF3{0, 0, 0};

    TF3 domain_min, domain_max;
    calculate_domain_and_resolution(*distance, margin, cell_width, sign, *cn_,
                                    domain_min_list_[i], domain_max_list_[i],
                                    resolution);
    distance_list_[i].reset(distance);
    resolution_list_[i] = resolution;
    sign_list_[i] = sign;

    // placeholders
    store_.remove(*distance_grids_[i]);
    distance_grids_[i].reset(store_.create<1, TF>({0}));
    store_.remove(*volume_grids_[i]);
    volume_grids_[i].reset(store_.create<1, TF>({0}));
    grid_size_list_[i] = 0;
    cell_size_list_[i] = TF3{0, 0, 0};

    v_(i) = TF3{0, 0, 0};
    x_(i) = x;
    q_(i) = q;
    omega_(i) = TF3{0, 0, 0};

    MeshBuffer mesh_buffer;
    if (store_.has_display()) {
      mesh_buffer = store_.get_display()->create_mesh_buffer(display_mesh);
      store_.get_display()->remove_mesh_buffer(mesh_buffer_list_[i]);
    }
    mesh_buffer_list_[i] = mesh_buffer;

    Variable<1, TF3>* collision_vertices_var =
        store_.create<1, TF3>({static_cast<U>(pellets.get_linear_shape())});
    if (volume_method_ == VolumeMethod::pellets) {
      num_pellets_ -= collision_vertex_list_[i]->get_linear_shape();
    }
    num_collision_pellets_ -= collision_vertex_list_[i]->get_linear_shape();
    store_.remove(*collision_vertex_list_[i]);
    collision_vertex_list_[i].reset(collision_vertices_var);
    collision_vertices_var->set_from(pellets);
    if (volume_method_ == VolumeMethod::pellets) {
      num_pellets_ += collision_vertices_var->get_linear_shape();
      if (num_pellets_ > max_num_pellets_) {
        std::stringstream error_sstream;
        error_sstream << "Cannot add pellets because the new pellet count "
                      << num_pellets_ << " exceeds the limit "
                      << max_num_pellets_;
        throw std::runtime_error(error_sstream.str());
      }
    }
    num_collision_pellets_ += collision_vertices_var->get_linear_shape();
    store_.remove(*pellet_id_to_rigid_id_);
    pellet_id_to_rigid_id_.reset(store_.create<1, U>({num_collision_pellets_}));

    build_grid(i);
  }
  void replace(U i, dg::Distance<TF3, TF>* distance, U3 resolution = U3{0},
               TF cell_width = 0,
               TF margin = std::numeric_limits<TF>::infinity(), TF sign = 1,
               Mesh const& collision_mesh = Mesh(), TF mass = 0,
               TF restitution = 1, TF friction = 0,
               TF3 const& inertia_tensor = TF3{1, 1, 1}, TF3 const& x = TF3{0},
               TQ const& q = TQ{0, 0, 0, 1},
               Mesh const& display_mesh = Mesh()) {
    mass_[i] = mass;
    restitution_[i] = restitution;
    friction_[i] = friction;
    inertia_tensor_[i] = inertia_tensor;

    a_[i] = TF3{0, 0, 0};
    force_[i] = TF3{0, 0, 0};
    torque_[i] = TF3{0, 0, 0};

    TF3 domain_min, domain_max;
    calculate_domain_and_resolution(*distance, margin, cell_width, sign, *cn_,
                                    domain_min_list_[i], domain_max_list_[i],
                                    resolution);
    distance_list_[i].reset(distance);
    resolution_list_[i] = resolution;
    sign_list_[i] = sign;

    // placeholders
    store_.remove(*distance_grids_[i]);
    distance_grids_[i].reset(store_.create<1, TF>({0}));
    store_.remove(*volume_grids_[i]);
    volume_grids_[i].reset(store_.create<1, TF>({0}));
    grid_size_list_[i] = 0;
    cell_size_list_[i] = TF3{0, 0, 0};

    v_(i) = TF3{0, 0, 0};
    x_(i) = x;
    q_(i) = q;
    omega_(i) = TF3{0, 0, 0};

    MeshBuffer mesh_buffer;
    if (store_.has_display()) {
      mesh_buffer = store_.get_display()->create_mesh_buffer(display_mesh);
      store_.get_display()->remove_mesh_buffer(mesh_buffer_list_[i]);
    }
    mesh_buffer_list_[i] = mesh_buffer;

    Variable<1, TF3>* collision_vertices_var =
        store_.create<1, TF3>({static_cast<U>(collision_mesh.vertices.size())});
    if (volume_method_ == VolumeMethod::pellets) {
      num_pellets_ -= collision_vertex_list_[i]->get_linear_shape();
    }
    num_collision_pellets_ -= collision_vertex_list_[i]->get_linear_shape();
    store_.remove(*collision_vertex_list_[i]);
    collision_vertex_list_[i].reset(collision_vertices_var);
    cast_vertices(*collision_vertices_var, collision_mesh.vertices);
    if (volume_method_ == VolumeMethod::pellets) {
      num_pellets_ += collision_vertices_var->get_linear_shape();
      if (num_pellets_ > max_num_pellets_) {
        std::stringstream error_sstream;
        error_sstream << "Cannot add pellets because the new pellet count "
                      << num_pellets_ << " exceeds the limit "
                      << max_num_pellets_;
        throw std::runtime_error(error_sstream.str());
      }
    }
    num_collision_pellets_ += collision_vertices_var->get_linear_shape();
    store_.remove(*pellet_id_to_rigid_id_);
    pellet_id_to_rigid_id_.reset(store_.create<1, U>({num_collision_pellets_}));

    build_grid(i);
  }
  void build_grid(U i) {
    U& num_nodes = grid_size_list_[i];
    TF3& cell_size = cell_size_list_[i];
    TF3& domain_min = domain_min_list_[i];
    TF3& domain_max = domain_max_list_[i];
    TF& sign = sign_list_[i];

    dg::Distance<TF3, TF> const& virtual_dist = *distance_list_[i];
    calculate_grid_attributes(virtual_dist, resolution_list_[i], domain_min,
                              domain_max, num_nodes, cell_size);

    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TBoxShellDistance = dg::BoxShellDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    using TInfiniteTubeDistance = dg::InfiniteTubeDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      store_.remove(*distance_grids_[i]);
      Variable<1, TF>* distance_grid = store_.create<1, TF>({num_nodes});
      distance_grids_[i].reset(distance_grid);
      std::vector<TF> nodes_host =
          construct_distance_grid(virtual_dist, resolution_list_[i],
                                  sign_list_[i], domain_min, domain_max);
      distance_grid->set_bytes(nodes_host.data());
    }

    if (volume_method_ == VolumeMethod::volume_map) {
      store_.remove(*volume_grids_[i]);
      Variable<1, TF>* volume_grid = store_.create<1, TF>({num_nodes});
      volume_grids_[i].reset(volume_grid);
      volume_grid->set_zero();
      if (TMeshDistance const* distance =
              dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance_grids_[i], domain_min, domain_max,
                  resolution_list_[i], cell_size, num_nodes, sign_list_[i]);
            },
            "update_volume_field", update_volume_field<TF3, TF>);
      } else if (TBoxDistance const* distance =
                     dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, sign_list_[i]);
            },
            "update_volume_field(BoxDistance)",
            update_volume_field<TF3, TF, TBoxDistance>);
      } else if (TBoxShellDistance const* distance =
                     dynamic_cast<TBoxShellDistance const*>(&virtual_dist)) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, sign_list_[i]);
            },
            "update_volume_field(BoxShellDistance)",
            update_volume_field<TF3, TF, TBoxShellDistance>);
      } else if (TSphereDistance const* distance =
                     dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, sign_list_[i]);
            },
            "update_volume_field(SphereDistance)",
            update_volume_field<TF3, TF, TSphereDistance>);
      } else if (TCylinderDistance const* distance =
                     dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, sign_list_[i]);
            },
            "update_volume_field(CylinderDistance)",
            update_volume_field<TF3, TF, TCylinderDistance>);
      } else if (TInfiniteCylinderDistance const* distance =
                     dynamic_cast<TInfiniteCylinderDistance const*>(
                         &virtual_dist)) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, sign_list_[i]);
            },
            "update_volume_field(InfiniteCylinderDistance)",
            update_volume_field<TF3, TF, TInfiniteCylinderDistance>);
      } else if (TInfiniteTubeDistance const* distance =
                     dynamic_cast<TInfiniteTubeDistance const*>(
                         &virtual_dist)) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, sign_list_[i]);
            },
            "update_volume_field(InfiniteTubeDistance)",
            update_volume_field<TF3, TF, TInfiniteTubeDistance>);
      } else if (TCapsuleDistance const* distance =
                     dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
        runner_.launch(
            num_nodes,
            [&](U grid_size, U block_size) {
              update_volume_field<<<grid_size, block_size>>>(
                  *volume_grid, *distance, domain_min, resolution_list_[i],
                  cell_size, num_nodes, sign_list_[i]);
            },
            "update_volume_field(CapsuleDistance)",
            update_volume_field<TF3, TF, TCapsuleDistance>);
      } else {
        std::stringstream error_sstream;
        error_sstream << "[update_volume_field] Distance type not supported.";
        std::cerr << error_sstream.str() << std::endl;
      }
    }
  }
  void update_pellet_id_to_rigid_id() {
    for (U i = 0; i < get_size(); ++i) {
      update_pellet_id_to_rigid_id(i);
    }
  }
  void update_pellet_id_to_rigid_id(U i) {
    pellet_index_offset_list_[i] =
        i > 0 ? pellet_index_offset_list_[i - 1] +
                    collision_vertex_list_[i - 1]->get_linear_shape()
              : 0;
    pellet_id_to_rigid_id_->fill(i,
                                 collision_vertex_list_[i]->get_linear_shape(),
                                 pellet_index_offset_list_[i]);
  }
  void hint_identical_sequence(U begin_id, U end_id) {
    identical_sequence_list_.push_back(U2{begin_id, end_id});
  }
  void compute_custom_beads_internal(U i, Variable<1, U>& internal_encoded,
                                     Variable<1, TF3> const& bead_x) {
    runner_.launch_compute_custom_beads_internal(
        internal_encoded, bead_x, *distance_list_[i], *distance_grids_[i],
        domain_min_list_[i], domain_max_list_[i], resolution_list_[i],
        cell_size_list_[i], sign_list_[i], x_(i), q_(i),
        internal_encoded.get_linear_shape());
  }
  U compute_sort_custom_beads_internal_all(Variable<1, U>& internal_encoded,
                                           Variable<1, TF3> const& bead_x) {
    internal_encoded.set_zero();
    for (U i = 0; i < get_size(); ++i) {
      compute_custom_beads_internal(i, internal_encoded, bead_x);
    }
    TRunner::sort(internal_encoded, internal_encoded.get_linear_shape());
    return internal_encoded.get_linear_shape() -
           TRunner::count(internal_encoded, UINT_MAX,
                          internal_encoded.get_linear_shape());
  }
  void compute_fluid_block_internal(U i, Variable<1, U>& internal_encoded,
                                    TF3 const& box_min, TF3 const& box_max,
                                    TF particle_radius, int mode) {
    runner_.launch_compute_fluid_block_internal(
        internal_encoded, *distance_list_[i], *distance_grids_[i],
        domain_min_list_[i], domain_max_list_[i], resolution_list_[i],
        cell_size_list_[i], sign_list_[i], x_(i), q_(i),
        internal_encoded.get_linear_shape(), particle_radius, mode, box_min,
        box_max);
  }
  U compute_sort_fluid_block_internal_all(Variable<1, U>& internal_encoded,
                                          TF3 const& box_min,
                                          TF3 const& box_max,
                                          TF particle_radius, int mode) {
    internal_encoded.set_zero();
    for (U i = 0; i < get_size(); ++i) {
      compute_fluid_block_internal(i, internal_encoded, box_min, box_max,
                                   particle_radius, mode);
    }
    TRunner::sort(internal_encoded, internal_encoded.get_linear_shape());
    return internal_encoded.get_linear_shape() -
           TRunner::count(internal_encoded, UINT_MAX,
                          internal_encoded.get_linear_shape());
  }
  void compute_fluid_cylinder_internal(U i, Variable<1, U>& internal_encoded,
                                       TF radius, TF particle_radius, TF y_min,
                                       TF y_max) {
    runner_.launch_compute_fluid_cylinder_internal(
        internal_encoded, *distance_list_[i], *distance_grids_[i],
        domain_min_list_[i], domain_max_list_[i], resolution_list_[i],
        cell_size_list_[i], sign_list_[i], x_(i), q_(i),
        internal_encoded.get_linear_shape(), radius, particle_radius, y_min,
        y_max);
  }
  U compute_sort_fluid_cylinder_internal_all(Variable<1, U>& internal_encoded,
                                             TF radius, TF particle_radius,
                                             TF y_min, TF y_max) {
    internal_encoded.set_zero();
    for (U i = 0; i < get_size(); ++i) {
      compute_fluid_cylinder_internal(i, internal_encoded, radius,
                                      particle_radius, y_min, y_max);
    }
    TRunner::sort(internal_encoded, internal_encoded.get_linear_shape());
    return internal_encoded.get_linear_shape() -
           TRunner::count(internal_encoded, UINT_MAX,
                          internal_encoded.get_linear_shape());
  }
  void set_gravity(TF3 gravity) { gravity_ = gravity; }
  void reallocate_kinematics_on_device() {
    store_.remove(*x_device_);
    store_.remove(*v_device_);
    store_.remove(*q_device_);
    store_.remove(*omega_device_);
    store_.remove(*mass_contact_device_);
    store_.remove(*inertia_device_);
    x_device_.reset(store_.create<1, TF3>({get_size()}));
    v_device_.reset(store_.create<1, TF3>({get_size()}));
    q_device_.reset(store_.create<1, TQ>({get_size()}));
    omega_device_.reset(store_.create<1, TF3>({get_size()}));
    mass_contact_device_.reset(store_.create<1, TF3>({get_size()}));
    inertia_device_.reset(store_.create<1, TF3>({get_size()}));

    std::vector<TF3> mass_contact_host;
    mass_contact_host.reserve(get_size());
    for (U i = 0; i < get_size(); ++i) {
      mass_contact_host.push_back(TF3{mass_[i], restitution_[i], friction_[i]});
    }
    mass_contact_device_->set_bytes(mass_contact_host.data());
    inertia_device_->set_bytes(inertia_tensor_.data());

    cni_->num_boundaries = get_size();
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
    q_device_->set_bytes(q_.ptr_);
    omega_device_->set_bytes(omega_.ptr_);
  }
  void write_file(const char* filename, TF x_scale = 1, TF v_scale = 1,
                  TF omega_scale = 1) const {
    std::ofstream stream(filename, std::ios::binary | std::ios::trunc);
    for (U i = 0; i < get_size(); ++i) {
      TF3 x_scaled = x_(i) * x_scale;
      TF3 v_scaled = v_(i) * v_scale;
      TF3 omega_scaled = omega_(i) * omega_scale;
      stream.write(reinterpret_cast<const char*>(&x_scaled), sizeof(TF3));
      stream.write(reinterpret_cast<const char*>(&v_scaled), sizeof(TF3));
      stream.write(reinterpret_cast<const char*>(&q_(i)), sizeof(TQ));
      stream.write(reinterpret_cast<const char*>(&omega_scaled), sizeof(TF3));
    }
  }
  // TODO: rename to load
  void read_file(const char* filename, int num_rigids = -1, U dst_offset = 0,
                 U src_offset = 0) {
    std::ifstream stream(filename, std::ios::binary);
    U num_rigids_to_read = get_size();
    if (num_rigids >= 0) {
      num_rigids_to_read = static_cast<U>(num_rigids);
    }
    num_rigids_to_read = min(get_size(), num_rigids_to_read);
    stream.seekg((sizeof(TF3) * 3 + sizeof(TQ)) * src_offset,
                 std::ios_base::cur);
    for (U i = 0; i < num_rigids_to_read; ++i) {
      U dst_id = dst_offset + i;
      stream.read(reinterpret_cast<char*>(&x_(dst_id)), sizeof(TF3));
      stream.read(reinterpret_cast<char*>(&v_(dst_id)), sizeof(TF3));
      stream.read(reinterpret_cast<char*>(&q_(dst_id)), sizeof(TQ));
      stream.read(reinterpret_cast<char*>(&omega_(dst_id)), sizeof(TF3));
      if (stream.peek() == std::ifstream::traits_type::eof()) break;
    }
  }
  static U get_size_from_file(const char* filename) {
    std::ifstream stream(filename, std::ifstream::ate | std::ifstream::binary);
    return stream.tellg() / (sizeof(TF3) * 3 + sizeof(TQ));
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
  U find_contacts(U i, U j) {
    num_contacts_->set_zero();
    Variable<1, TF3>& vertices_i = *collision_vertex_list_[i];
    U num_vertices_i = vertices_i.get_linear_shape();
    runner_.launch_collision_test(
        *distance_list_[j], *distance_grids_[j], i, j, vertices_i,
        *num_contacts_, *contacts_, mass_[i], inertia_tensor_[i], x_(i), v_(i),
        q_(i), omega_(i), mass_[j], inertia_tensor_[j], x_(j), v_(j), q_(j),
        omega_(j), restitution_[i] * restitution_[j],
        friction_[i] + friction_[j], domain_min_list_[j], domain_max_list_[j],
        resolution_list_[j], cell_size_list_[j], sign_list_[j], num_vertices_i);
    num_contacts_->get_bytes(num_contacts_pinned_.ptr_);
    return num_contacts_pinned_(0);
  }
  U find_contacts(U i) {
    num_contacts_->set_zero();
    Variable<1, TF3>& vertices_i = *collision_vertex_list_[i];
    U num_vertices_i = vertices_i.get_linear_shape();
    for (U j = 0; j < get_size(); ++j) {
      if (i == j || (mass_[i] == 0 and mass_[j] == 0)) continue;
      runner_.launch_collision_test(
          *distance_list_[j], *distance_grids_[j], i, j, vertices_i,
          *num_contacts_, *contacts_, mass_[i], inertia_tensor_[i], x_(i),
          v_(i), q_(i), omega_(i), mass_[j], inertia_tensor_[j], x_(j), v_(j),
          q_(j), omega_(j), restitution_[i] * restitution_[j],
          friction_[i] + friction_[j], domain_min_list_[j], domain_max_list_[j],
          resolution_list_[j], cell_size_list_[j], sign_list_[j],
          num_vertices_i);
    }
    num_contacts_->get_bytes(num_contacts_pinned_.ptr_);
    return num_contacts_pinned_(0);
  }
  U find_contacts(Variable<1, TF3> const& particle_x, U max_num_beads) {
    num_contacts_->set_zero();
    if (volume_method_ == VolumeMethod::pellets) {
      U index_to_identical_sequence_list = 0;
      U j = 0;
      while (j < get_size()) {
        U j_begin = j;
        U j_end = j + 1;
        U2 identical_sequence{0};
        if (index_to_identical_sequence_list <
            identical_sequence_list_.size()) {
          identical_sequence =
              identical_sequence_list_[index_to_identical_sequence_list];
        }
        if (j == identical_sequence.x && j < identical_sequence.y) {
          j_begin = identical_sequence.x;
          j_end = identical_sequence.y;
          index_to_identical_sequence_list++;
        }
        runner_.launch_collision_test_with_pellets(
            *distance_list_[j], *distance_grids_[j], j_begin, j_end, particle_x,
            *pellet_id_to_rigid_id_, *num_contacts_, *contacts_, *x_device_,
            *v_device_, *q_device_, *omega_device_, *mass_contact_device_,
            *inertia_device_, domain_min_list_[j], domain_max_list_[j],
            resolution_list_[j], cell_size_list_[j], sign_list_[j],
            max_num_beads, num_pellets_);
        j = j_end;
      }
    } else {
      for (U i = 0; i < get_size(); ++i) {
        Variable<1, TF3>& vertices_i = *collision_vertex_list_[i];
        U num_vertices_i = vertices_i.get_linear_shape();
        for (U j = 0; j < get_size(); ++j) {
          if (i == j || (mass_[i] == 0 and mass_[j] == 0)) continue;
          runner_.launch_collision_test(
              *distance_list_[j], *distance_grids_[j], i, j, vertices_i,
              *num_contacts_, *contacts_, mass_[i], inertia_tensor_[i], x_(i),
              v_(i), q_(i), omega_(i), mass_[j], inertia_tensor_[j], x_(j),
              v_(j), q_(j), omega_(j), restitution_[i] * restitution_[j],
              friction_[i] + friction_[j], domain_min_list_[j],
              domain_max_list_[j], resolution_list_[j], cell_size_list_[j],
              sign_list_[j], num_vertices_i);
        }
      }
    }
    num_contacts_->get_bytes(num_contacts_pinned_.ptr_);
    return num_contacts_pinned_(0);
  }
  void solve_contacts() {
    U num_contacts = num_contacts_pinned_(0);
    if (num_contacts > cni_->max_num_contacts) {
      std::stringstream error_sstream;
      error_sstream << "No. of contacts exceeds " << cni_->max_num_contacts
                    << ".";
      std::cerr << error_sstream.str() << std::endl;
      num_contacts = cni_->max_num_contacts;
    }
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
  static void cast_vertices(Variable<1, TF3>& vertex_var,
                            VertexList const& vertices) {
    if constexpr (std::is_same_v<TF3, float3>) {
      vertex_var.set_bytes(vertices.data());
    } else if constexpr (std::is_same_v<TF3, double3>) {
      std::vector<TF3> converted_vertices;
      converted_vertices.reserve(vertices.size());
      for (float3 const& vertex : vertices) {
        converted_vertices.push_back(TF3{vertex.x, vertex.y, vertex.z});
      }
      vertex_var.set_bytes(converted_vertices.data());
    }
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
  void for_rigid(U i, Lambda f) {
    f(i, *distance_list_[i], *collision_vertex_list_[i],
      pellet_index_offset_list_[i], *distance_grids_[i], *volume_grids_[i],
      x_(i), v_(i), q_(i), omega_(i), domain_min_list_[i], domain_max_list_[i],
      resolution_list_[i], cell_size_list_[i], grid_size_list_[i],
      sign_list_[i]);
  }
  template <class Lambda>
  void for_each_rigid(Lambda f) {
    for (U i = 0; i < get_size(); ++i) {
      for_rigid(i, f);
    }
  }

  void compute_mask(U i, TF distance_threshold,
                    Variable<1, TF3> const& sample_x, Variable<1, TF>& mask,
                    U num_samples) {
    for_rigid(
        i,
        [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
            Variable<1, TF3> const& local_pellet_x, U pellet_index_offset,
            Variable<1, TF> const& distance_grid,
            Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
            TF3 const& rigid_v, TQ const& rigid_q, TF3 const& rigid_omega,
            TF3 const& domain_min, TF3 const& domain_max, U3 const& resolution,
            TF3 const& cell_size, U num_nodes, TF sign) {
          runner_.launch_compute_boundary_mask(
              distance, distance_grid, rigid_x, rigid_q, domain_min, domain_max,
              resolution, cell_size, sign, sample_x, distance_threshold, mask,
              num_samples);
        });
  }
};
}  // namespace alluvion

#endif /* ALLUVION_PILE_HPP */
