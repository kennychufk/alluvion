#ifndef ALLUVION_PILE_HPP
#define ALLUVION_PILE_HPP

#include <memory>

#include "alluvion/dg/cubic_lagrange_discrete_grid.hpp"
#include "alluvion/dg/mesh_distance.hpp"
#include "alluvion/store.hpp"

namespace alluvion {
class Pile {
 private:
  static dg::MeshDistance* construct_mesh_distance(VertexList const& vertices,
                                                   FaceList const& faces);
  static std::vector<F> construct_distance_grid(dg::Distance const& distance,
                                                U3 const& resolution, F margin,
                                                F sign, F thickness,
                                                F3& domain_min, F3& domain_max,
                                                U& grid_size, F3& cell_size);

 public:
  std::vector<F> mass_;
  std::vector<F> restitution_;
  std::vector<F> friction_;
  std::vector<F> boundary_viscosity_;
  std::vector<F3> inertia_tensor_;

  std::vector<F3> x_mat_;
  std::vector<Q> q_mat_;
  std::vector<Q> q_initial_;

  PinnedVariable<1, F3> x_;
  std::vector<F3> oldx_;
  PinnedVariable<1, F3> v_;
  std::vector<F3> a_;
  std::vector<F3> force_;

  std::vector<Q> q_;
  PinnedVariable<1, F3> omega_;
  std::vector<F3> torque_;

  std::vector<std::unique_ptr<dg::Distance>> distance_list_;
  std::vector<U3> resolution_list_;
  std::vector<F> sign_list_;
  std::vector<F> thickness_list_;
  std::vector<Variable<1, F>> distance_grids_;
  std::vector<Variable<1, F>> volume_grids_;
  std::vector<F3> domain_min_list_;
  std::vector<F3> domain_max_list_;
  std::vector<U> grid_size_list_;
  std::vector<F3> cell_size_list_;

  std::vector<MeshBuffer> mesh_buffer_list_;
  std::vector<Variable<1, F3>> collision_vertex_list_;

  Variable<1, F3> x_device_;
  Variable<1, F3> v_device_;
  Variable<1, F3> omega_device_;
  Variable<1, F> boundary_viscosity_device_;
  Variable<1, Contact> contacts_;
  Variable<1, U> num_contacts_;
  PinnedVariable<1, Contact> contacts_pinned_;
  PinnedVariable<1, U> num_contacts_pinned_;
  F3 gravity_;

  Store& store_;

  Pile(Store& store, U max_num_contacts);
  virtual ~Pile();
  void add(Mesh const& field_mesh, U3 const& resolution, F sign, F thickness,
           Mesh const& collision_mesh, F mass, F restitution, F friction,
           F boundary_viscosity, F3 const& inertia_tensor, F3 const& x,
           Q const& q, Mesh const& display_mesh);
  void add(dg::Distance* distance, U3 const& resolution, F sign, F thickness,
           Mesh const& collision_mesh, F mass, F restitution, F friction,
           F boundary_viscosity, F3 const& inertia_tensor, F3 const& x,
           Q const& q, Mesh const& display_mesh);
  void build_grids(F margin);
  void set_gravity(F3 gravity);
  void reallocate_kinematics_on_device();
  void reallocate_kinematics_on_pinned();
  void copy_kinematics_to_device();
  void integrate_kinematics(F dt);
  void find_contacts();
  void solve_contacts();
  U get_size() const;
  glm::mat4 get_matrix(U i) const;

  template <class Lambda>
  void for_each_rigid(Lambda f) {
    for (U i = 0; i < get_size(); ++i) {
      f(i, distance_grids_[i], volume_grids_[i], x_(i), q_[i],
        domain_min_list_[i], domain_max_list_[i], resolution_list_[i],
        cell_size_list_[i], grid_size_list_[i], sign_list_[i],
        thickness_list_[i]);
    }
  }
};
}  // namespace alluvion

#endif /* ALLUVION_PILE_HPP */
