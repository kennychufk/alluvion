#ifndef ALLUVION_SOLVER_HPP
#define ALLUVION_SOLVER_HPP

#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/usher.hpp"
namespace alluvion {
template <typename TF>
struct Solver {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  Solver(TRunner& runner_arg, TPile& pile_arg, Store& store_arg,
         U max_num_particles_arg, U3 grid_res, U num_ushers = 0,
         bool enable_surface_tension_arg = false,
         bool enable_vorticity_arg = false, bool graphical = false)
      : max_num_particles(max_num_particles_arg),
        store(store_arg),
        runner(runner_arg),
        pile(pile_arg),
        usher(new Usher<TF>(store_arg, num_ushers)),
        dt(0),
        t(0),
        particle_radius(store_arg.get_cn<TF>().particle_radius),
        next_emission_t(-1),
        particle_x(
            graphical
                ? store_arg.create_graphical<1, TF3>({max_num_particles_arg})
                : store_arg.create<1, TF3>({max_num_particles_arg})),
        particle_v(store_arg.create<1, TF3>({max_num_particles_arg})),
        particle_a(store_arg.create<1, TF3>({max_num_particles_arg})),
        particle_density(store_arg.create<1, TF>({max_num_particles_arg})),
        particle_boundary(
            store_arg.create<2, TQ>({pile.get_size(), max_num_particles_arg})),
        particle_boundary_kernel(
            store_arg.create<2, TQ>({pile.get_size(), max_num_particles_arg})),
        particle_force(
            store_arg.create<2, TF3>({pile.get_size(), max_num_particles_arg})),
        particle_torque(
            store_arg.create<2, TF3>({pile.get_size(), max_num_particles_arg})),
        particle_cfl_v2(store_arg.create<1, TF>({max_num_particles_arg})),
        particle_normal(enable_surface_tension_arg
                            ? store_arg.create<1, TF3>({max_num_particles_arg})
                            : new Variable<1, TF3>()),
        particle_omega(enable_vorticity_arg
                           ? store_arg.create<1, TF3>({max_num_particles_arg})
                           : new Variable<1, TF3>()),
        particle_angular_acceleration(
            enable_vorticity_arg
                ? store_arg.create<1, TF3>({max_num_particles_arg})
                : new Variable<1, TF3>()),
        pid(store_arg.create<4, TQ>(
            {grid_res.x, grid_res.y, grid_res.z,
             store_arg.get_cni().max_num_particles_per_cell})),
        pid_length(
            store_arg.create<3, U>({grid_res.x, grid_res.y, grid_res.z})),
        particle_neighbors(store_arg.create<2, TQ>(
            {max_num_particles_arg,
             store_arg.get_cni().max_num_neighbors_per_particle})),
        particle_num_neighbors(store_arg.create<1, U>({max_num_particles_arg})),
        enable_surface_tension(enable_surface_tension_arg),
        enable_vorticity(enable_vorticity_arg) {}
  virtual ~Solver() {
    store.remove(*particle_x);
    store.remove(*particle_v);
    store.remove(*particle_a);
    store.remove(*particle_density);
    store.remove(*particle_boundary);
    store.remove(*particle_boundary_kernel);
    store.remove(*particle_force);
    store.remove(*particle_torque);
    store.remove(*particle_cfl_v2);
    store.remove(*particle_normal);
    store.remove(*particle_omega);
    store.remove(*particle_angular_acceleration);
    store.remove(*pid);
    store.remove(*pid_length);
    store.remove(*particle_neighbors);
    store.remove(*particle_num_neighbors);
  }
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
        "scale", scale<TF>);
  }
  virtual void reset_solving_var() {
    if (enable_vorticity) {
      particle_omega->set_zero();
    }
  }
  void emit_single(TF3 const& x, TF3 const& v) {
    if (t < next_emission_t || num_particles == max_num_particles) return;
    particle_x->set_bytes(&x, sizeof(TF3), sizeof(TF3) * num_particles);
    particle_v->set_bytes(&v, sizeof(TF3), sizeof(TF3) * num_particles);
    next_emission_t = t + particle_radius * 2 / length(v);
    ++num_particles;
  }
  void emit_circle(TF3 const& center, TF3 const& v, TF radius, U num_emission) {
    if (t < next_emission_t || num_particles == max_num_particles) return;
    num_emission = min(num_emission, max_num_particles - num_particles);
    runner.launch(
        num_emission,
        [&](U grid_size, U block_size) {
          emit_cylinder_sunflower<<<grid_size, block_size>>>(
              *particle_x, *particle_v, num_emission, num_particles, radius,
              center, v);
        },
        "emit_cylinder_sunflower", emit_cylinder_sunflower<TF3, TF>);
    next_emission_t = t + particle_radius * 2 / length(v);
    num_particles += num_emission;
  }
  void move_particles_naive(TF3 const& exclusion_min,
                            TF3 const& exclusion_max) {
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          move_particles<<<grid_size, block_size>>>(
              *particle_x, *particle_v, dt, exclusion_min, exclusion_max,
              num_particles);
        },
        "move_particles", move_particles<TF3, TF>);
  }
  void dictate_ethier_steinman(TF a, TF d, TF kinematic_viscosity,
                               TF3 const& exclusion_min,
                               TF3 const& exclusion_max) {
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          set_ethier_steinman<<<grid_size, block_size>>>(
              *particle_x, *particle_v, a, d, kinematic_viscosity, t,
              exclusion_min, exclusion_max, num_particles);
        },
        "set_ethier_steinman", set_ethier_steinman<TF3, TF>);
  }
  void set_mask(Variable<1, U>& mask, TF3 const& box_min, TF3 const& box_max) {
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          set_box_mask<<<grid_size, block_size>>>(*particle_x, mask, box_min,
                                                  box_max, num_particles);
        },
        "set_box_mask", set_box_mask<TF3>);
  }
  template <U wrap>
  void update_particle_neighbors() {
    pid_length->set_zero();
    runner.launch_update_particle_grid(*particle_x, *pid, *pid_length,
                                       num_particles);

    runner.template launch_make_neighbor_list<wrap>(
        *particle_x, *pid, *pid_length, *particle_neighbors,
        *particle_num_neighbors, num_particles);

    runner.launch_compute_density(*particle_x, *particle_neighbors,
                                  *particle_num_neighbors, *particle_density,
                                  *particle_boundary_kernel, num_particles);
  }
  void sample_usher() {
    runner.template launch_make_neighbor_list<0>(
        *usher->sample_x, *pid, *pid_length, *usher->neighbors,
        *usher->num_neighbors, usher->num_ushers);
    runner.launch_sample_fluid(*usher->sample_x, *particle_x, *particle_density,
                               *particle_density, *usher->neighbors,
                               *usher->num_neighbors, *usher->sample_density,
                               usher->num_ushers);
    runner.launch_sample_fluid(*usher->sample_x, *particle_x, *particle_density,
                               *particle_v, *usher->neighbors,
                               *usher->num_neighbors, *usher->sample_v,
                               usher->num_ushers);
  }
  U max_num_particles;
  U num_particles;
  TF t;
  TF dt;
  TF max_dt;
  TF min_dt;
  TF cfl;
  TF particle_radius;
  bool enable_surface_tension;
  bool enable_vorticity;
  TF next_emission_t;

  Store& store;
  TRunner& runner;
  TPile& pile;
  std::unique_ptr<Usher<TF>> usher;

  std::unique_ptr<Variable<1, TF3>> particle_x;
  std::unique_ptr<Variable<1, TF3>> particle_v;
  std::unique_ptr<Variable<1, TF3>> particle_a;
  std::unique_ptr<Variable<1, TF>> particle_density;
  std::unique_ptr<Variable<2, TQ>> particle_boundary;
  std::unique_ptr<Variable<2, TQ>> particle_boundary_kernel;
  std::unique_ptr<Variable<2, TF3>> particle_force;
  std::unique_ptr<Variable<2, TF3>> particle_torque;
  std::unique_ptr<Variable<1, TF>> particle_cfl_v2;
  std::unique_ptr<Variable<1, TF3>> particle_normal;
  std::unique_ptr<Variable<1, TF3>> particle_omega;
  std::unique_ptr<Variable<1, TF3>> particle_angular_acceleration;

  std::unique_ptr<Variable<4, TQ>> pid;
  std::unique_ptr<Variable<3, U>> pid_length;
  std::unique_ptr<Variable<2, TQ>> particle_neighbors;
  std::unique_ptr<Variable<1, U>> particle_num_neighbors;
};
}  // namespace alluvion

#endif
