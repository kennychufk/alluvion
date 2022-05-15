#ifndef ALLUVION_SOLVER_PELLET_HPP
#define ALLUVION_SOLVER_PELLET_HPP

#include "alluvion/solver.hpp"
#include "alluvion/store.hpp"
namespace alluvion {
template <typename TF>
struct SolverPellet : public Solver<TF> {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  using Base = Solver<TF>;
  using Base::compute_all_boundaries;
  using Base::dt;
  using Base::num_particles;
  using Base::particle_radius;
  using Base::pile;
  using Base::runner;
  using Base::store;
  using Base::t;
  using Base::update_dt;

  using Base::particle_a;
  using Base::particle_angular_acceleration;
  using Base::particle_boundary;
  using Base::particle_boundary_kernel;
  using Base::particle_cfl_v2;
  using Base::particle_density;
  using Base::particle_normal;
  using Base::particle_omega;
  using Base::particle_v;
  using Base::particle_x;

  using Base::particle_boundary_neighbors;
  using Base::particle_neighbors;
  using Base::particle_num_boundary_neighbors;
  using Base::particle_num_neighbors;
  using Base::pid;
  using Base::pid_length;

  // NOTE: pile should use VolumeMethod::pellets but contains no pellets and
  // contains one object only
  SolverPellet(TRunner& runner_arg, TPile& pile_arg, Store& store_arg,
               U max_num_particles_arg, bool graphical = false)
      : Base(runner_arg, pile_arg, store_arg, max_num_particles_arg, 0, false,
             false, graphical),
        particle_dfsph_factor(store_arg.create<1, TF>({max_num_particles_arg})),
        cohesion(4),
        adhesion(2) {}
  virtual ~SolverPellet() { store.remove(*particle_dfsph_factor); }
  void set_pellets(Variable<1, TF3> const& x) {
    num_particles = x.get_linear_shape();
    particle_x->set_from(x);
  }
  template <U wrap>
  void step() {
    pile.copy_kinematics_to_device();
    Base::template update_particle_neighbors<wrap>();

    pile.for_each_rigid(
        [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
            Variable<1, TF3> const& local_pellet_x, U pellet_index_offset,
            Variable<1, TF> const& distance_grid,
            Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
            TF3 const& rigid_v, TQ const& rigid_q, TF3 const& rigid_omega,
            TF3 const& domain_min, TF3 const& domain_max, U3 const& resolution,
            TF3 const& cell_size, U num_nodes, TF sign) {
          runner.launch_compute_cohesion_adhesion_displacement(
              *particle_x, *particle_density, *particle_neighbors,
              *particle_num_neighbors, *particle_v, distance, distance_grid,
              domain_min, domain_max, resolution, cell_size, sign, cohesion,
              adhesion, num_particles);
        });

    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_dfsph_factor_with_pellets<<<grid_size, block_size>>>(
              *particle_x, *particle_neighbors, *particle_num_neighbors,
              *particle_boundary_neighbors, *particle_num_boundary_neighbors,
              *particle_dfsph_factor, num_particles);
        },
        "compute_dfsph_factor_with_pellets",
        compute_dfsph_factor_with_pellets<TQ, TF3, TF>);

    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          pellet_divergence_solve_iteration<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_density,
              *particle_dfsph_factor, *particle_cfl_v2, *particle_neighbors,
              *particle_num_neighbors, num_particles);
        },
        "pellet_divergence_solve_iteration",
        pellet_divergence_solve_iteration<TQ, TF3, TF>);
    TF particle_max_v2 = TRunner::max(*particle_cfl_v2, num_particles);
    TF cfl_factor = 0.25;  // TODO: make a member variable
    TF cfl = min(cfl_factor * static_cast<TF>(0.4) * particle_radius * 2 *
                     rsqrt(particle_max_v2),
                 static_cast<TF>(1));

    pile.for_each_rigid(
        [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
            Variable<1, TF3> const& local_pellet_x, U pellet_index_offset,
            Variable<1, TF> const& distance_grid,
            Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
            TF3 const& rigid_v, TQ const& rigid_q, TF3 const& rigid_omega,
            TF3 const& domain_min, TF3 const& domain_max, U3 const& resolution,
            TF3 const& cell_size, U num_nodes, TF sign) {
          runner.template launch_move_and_avoid_boundary<wrap>(
              *particle_x, *particle_v, distance, distance_grid, domain_min,
              domain_max, resolution, cell_size, sign, cfl, num_particles);
        });
  }

  std::unique_ptr<Variable<1, TF>> particle_dfsph_factor;
  TF cohesion;
  TF adhesion;
};
}  // namespace alluvion

#endif
