#ifndef ALLUVION_PILE_HPP
#define ALLUVION_PILE_HPP

#include "alluvion/dg/cubic_lagrange_discrete_grid.hpp"
#include "alluvion/dg/mesh_distance.hpp"
#include "alluvion/store.hpp"

namespace alluvion {
class Pile {
 public:
  using VertexList = std::vector<F3>;
  using FaceList = std::vector<U3>;

 private:
  static dg::MeshDistance construct_mesh_distance(VertexList const& vertices,
                                                  FaceList const& faces,
                                                  F3& aabb_min, F3& aabb_max);
  static std::vector<F> construct_distance_grid(
      dg::MeshDistance const& mesh_distance, U3 const& resolution,
      F3 const& aabb_min, F3 const& aabb_max, F margin, F sign, F thickness,
      F3& domain_min, F3& domain_max, U& grid_size, F3& cell_size);
  static F find_max_distance(VertexList const& vertices);

 public:
  std::vector<F> mass_;
  std::vector<F> restitution_;
  std::vector<F> friction_;
  std::vector<F> boundary_viscosity_;
  std::vector<F3> inertia_tensor_;

  std::vector<F> max_dist_;
  std::vector<F3> x_mat_;
  std::vector<Q> q_mat_;
  std::vector<Q> q_initial_;

  std::vector<F3> x_;
  std::vector<F3> oldx_;
  std::vector<F3> v_;
  std::vector<F3> a_;
  std::vector<F3> force_;

  std::vector<Q> q_;
  std::vector<F3> omega_;
  std::vector<F3> torque_;

  std::vector<F3> aabb_min_list_;
  std::vector<F3> aabb_max_list_;
  std::vector<dg::MeshDistance> mesh_distance_list_;
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

  Store& store_;

  Pile(Store& store);
  virtual ~Pile();
  void add(VertexList const& field_vertices, FaceList const& field_faces,
           U3 const& resolution, F sign, F thickness,
           VertexList const& collision_vertices, F mass, F restitution,
           F friction, F boundary_viscosity, F3 const& inertia_tensor,
           F3 const& x, Q const& q, VertexList const& display_vertices,
           FaceList const& display_faces);
  void add(const char* field_mesh_filename, U3 const& resolution, F sign,
           F thickness, const char* collision_mesh_filename, F mass,
           F restitution, F friction, F boundary_viscosity,
           F3 const& inertia_tensor, F3 const& x, Q const& q,
           const char* display_mesh_filename);
  U get_size() const;
  void build_grids(F margin);
  void copy_kinematics_to_device();
  glm::mat4 get_matrix(U i) const;

  static void read_obj(const char* filename, VertexList* vertices,
                       FaceList* faces);
  template <class Lambda>
  void for_each_rigid(Lambda f) {
    for (U i = 0; i < get_size(); ++i) {
      f(i, distance_grids_[i], volume_grids_[i], x_[i], q_[i],
        domain_min_list_[i], domain_max_list_[i], resolution_list_[i],
        cell_size_list_[i], grid_size_list_[i], sign_list_[i],
        thickness_list_[i]);
    }
  }
};
}  // namespace alluvion

#endif /* ALLUVION_PILE_HPP */
