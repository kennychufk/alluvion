#ifndef ALLUVION_SOLVER_HPP
#define ALLUVION_SOLVER_HPP

#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
namespace alluvion {
template <typename TF>
struct Solver {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  Solver(TRunner& runner_arg, TPile& pile_arg, Store& store,
         U max_num_particles, U3 grid_res, U max_num_particles_per_cell = 64,
         U max_num_neighbors_per_particle = 64,
         bool enable_surface_tension_arg = false,
         bool enable_vorticity_arg = false, bool graphical = false)
      : runner(runner_arg),
        pile(pile_arg),
        dt(0),
        t(0),
        particle_x(graphical
                       ? store.create_graphical<1, TF3>({max_num_particles})
                       : store.create<1, TF3>({max_num_particles})),
        particle_v(store.create<1, TF3>({max_num_particles})),
        particle_a(store.create<1, TF3>({max_num_particles})),
        particle_density(store.create<1, TF>({max_num_particles})),
        particle_boundary_xj(
            store.create<2, TF3>({pile.get_size(), max_num_particles})),
        particle_boundary_volume(
            store.create<2, TF>({pile.get_size(), max_num_particles})),
        particle_force(
            store.create<2, TF3>({pile.get_size(), max_num_particles})),
        particle_torque(
            store.create<2, TF3>({pile.get_size(), max_num_particles})),
        particle_cfl_v2(store.create<1, TF>({max_num_particles})),
        particle_normal(enable_surface_tension_arg
                            ? store.create<1, TF3>({max_num_particles})
                            : new Variable<1, TF3>()),
        pid(store.create<4, TQ>(
            {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell})),
        pid_length(store.create<3, U>({grid_res.x, grid_res.y, grid_res.z})),
        particle_neighbors(store.create<2, TQ>(
            {max_num_particles, max_num_neighbors_per_particle})),
        particle_num_neighbors(store.create<1, U>({max_num_particles}))

  {}
  void normalize(Variable<1, TF3> const* v,
                 Variable<1, TF>* particle_normalized_attr, TF lower_bound,
                 TF upper_bound) {
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          normalize_vector_magnitude<<<grid_size, block_size>>>(
              *v, *particle_normalized_attr, lower_bound, upper_bound,
              num_particles);
        },
        "normalize_vector_magnitude", normalize_vector_magnitude<TF3, TF>);
  }
  void normalize(Variable<1, TF> const* s,
                 Variable<1, TF>* particle_normalized_attr, TF lower_bound,
                 TF upper_bound) {
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          scale<<<grid_size, block_size>>>(*s, *particle_normalized_attr,
                                           lower_bound, upper_bound,
                                           num_particles);
        },
        "normalize_vector_magnitude", normalize_vector_magnitude<TF3, TF>);
  }
  virtual void reset_solving_var() {}
  U num_particles;
  TF t;
  TF dt;
  TF max_dt;
  TF min_dt;
  TF cfl;
  TF particle_radius;
  bool enable_surface_tension;
  bool enable_vorticity;

  TRunner& runner;
  TPile& pile;

  std::unique_ptr<Variable<1, TF3>> particle_x;
  std::unique_ptr<Variable<1, TF3>> particle_v;
  std::unique_ptr<Variable<1, TF3>> particle_a;
  std::unique_ptr<Variable<1, TF>> particle_density;
  std::unique_ptr<Variable<2, TF3>> particle_boundary_xj;
  std::unique_ptr<Variable<2, TF>> particle_boundary_volume;
  std::unique_ptr<Variable<2, TF3>> particle_force;
  std::unique_ptr<Variable<2, TF3>> particle_torque;
  std::unique_ptr<Variable<1, TF>> particle_cfl_v2;
  std::unique_ptr<Variable<1, TF3>> particle_normal;

  std::unique_ptr<Variable<4, TQ>> pid;
  std::unique_ptr<Variable<3, U>> pid_length;
  std::unique_ptr<Variable<2, TQ>> particle_neighbors;
  std::unique_ptr<Variable<1, U>> particle_num_neighbors;
};
}  // namespace alluvion

#endif
