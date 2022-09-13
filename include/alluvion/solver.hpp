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
         U max_num_particles_arg, U num_ushers = 0,
         bool enable_surface_tension_arg = false,
         bool enable_vorticity_arg = false, Const<TF> const* cn = nullptr,
         ConstiN const* cni = nullptr, bool graphical = false)
      : max_num_particles(max_num_particles_arg),
        store(store_arg),
        runner(runner_arg),
        pile(pile_arg),
        usher(new Usher<TF>(store_arg, pile_arg, num_ushers)),
        initial_dt(0),
        dt(0),
        t(0),
        particle_radius(cn == nullptr ? store_arg.get_cn<TF>().particle_radius
                                      : cn->particle_radius),
        next_emission_t(-1),
        particle_x(
            graphical ? store_arg.create_graphical<1, TF3>(
                            {max_num_particles_arg + pile_arg.max_num_pellets_})
                      : store_arg.create<1, TF3>({max_num_particles_arg +
                                                  pile_arg.max_num_pellets_})),
        particle_v(store_arg.create<1, TF3>(
            {max_num_particles_arg + pile_arg.max_num_pellets_})),
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
        pid(cni == nullptr
                ? store_arg.create<4, TQ>(
                      {store_arg.get_cni().grid_res.x,
                       store_arg.get_cni().grid_res.y,
                       store_arg.get_cni().grid_res.z,
                       store_arg.get_cni().max_num_particles_per_cell})
                : store_arg.create<4, TQ>({cni->grid_res.x, cni->grid_res.y,
                                           cni->grid_res.z,
                                           cni->max_num_particles_per_cell})),
        pid_length(
            cni == nullptr
                ? store_arg.create<3, U>({store_arg.get_cni().grid_res.x,
                                          store_arg.get_cni().grid_res.y,
                                          store_arg.get_cni().grid_res.z})
                : store_arg.create<3, U>(
                      {cni->grid_res.x, cni->grid_res.y, cni->grid_res.z})),
        particle_neighbors(store_arg.create<2, TQ>(
            {max_num_particles_arg + pile_arg.max_num_pellets_,
             cni == nullptr ? store_arg.get_cni().max_num_neighbors_per_particle
                            : cni->max_num_neighbors_per_particle})),
        particle_num_neighbors(store_arg.create<1, U>(
            {max_num_particles_arg + pile_arg.max_num_pellets_})),
        particle_boundary_neighbors(store_arg.create<2, TQ>(
            {max_num_particles_arg + pile_arg.max_num_pellets_,
             cni == nullptr ? store_arg.get_cni().max_num_neighbors_per_particle
                            : cni->max_num_neighbors_per_particle})),
        particle_num_boundary_neighbors(store_arg.create<1, U>(
            {max_num_particles_arg + pile_arg.max_num_pellets_})),
        grid_anomaly(store_arg.create<1, U>({3})),
        enable_surface_tension(enable_surface_tension_arg),
        enable_vorticity(enable_vorticity_arg) {
    if (cn == nullptr) {
      store.copy_cn<TF>();
    } else {
      store.copy_cn_external(*cn, *cni);
    }
  }
  virtual ~Solver() {
    if (GraphicalVariable<1, TF3>* graphical_particle_x =
            dynamic_cast<GraphicalVariable<1, TF3>*>(particle_x.get())) {
      store.remove_graphical(*graphical_particle_x);
    } else {
      store.remove(*particle_x);
    }
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
    store.remove(*particle_boundary_neighbors);
    store.remove(*particle_num_boundary_neighbors);
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
  virtual void reset_t() {
    t = 0;
    dt = initial_dt;
  }
  virtual void update_dt() {
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          calculate_cfl_v2<<<grid_size, block_size>>>(
              *particle_v, *particle_a, *particle_cfl_v2, dt, num_particles);
        },
        "calculate_cfl_v2", calculate_cfl_v2<TF3, TF>);
    particle_max_v2 = TRunner::max(*particle_cfl_v2, num_particles);
    pile_max_v2 = pile.calculate_cfl_v2();
    max_v2 = max(particle_max_v2, pile_max_v2);
    TF max_v = sqrt(max_v2);
    TF length_scale = particle_radius * 2;
    cfl_dt = cfl * length_scale / max_v;
    dt = max(min(cfl_dt, max_dt), min_dt);
    utilized_cfl = dt * max_v / length_scale;
  }
  void compute_all_boundaries() {
    pile.for_each_rigid(
        [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
            Variable<1, TF3> const& local_pellet_x, U pellet_index_offset,
            Variable<1, TF> const& distance_grid,
            Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
            TF3 const& rigid_v, TQ const& rigid_q, TF3 const& rigid_omega,
            TF3 const& domain_min, TF3 const& domain_max, U3 const& resolution,
            TF3 const& cell_size, U num_nodes, TF sign) {
          runner.launch_compute_particle_boundary(
              distance, volume_grid, distance_grid, rigid_x, rigid_q,
              boundary_id, domain_min, domain_max, resolution, cell_size, sign,
              *particle_x, *particle_boundary, *particle_boundary_kernel,
              num_particles);
        });
  }
  void sample_all_boundaries(Variable<1, TF3>& sample_x,
                             Variable<2, TQ>& sample_boundary,
                             Variable<2, TQ>& sample_boundary_kernel,
                             U num_samples) {
    pile.for_each_rigid(
        [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
            Variable<1, TF3> const& local_pellet_x, U pellet_index_offset,
            Variable<1, TF> const& distance_grid,
            Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
            TF3 const& rigid_v, TQ const& rigid_q, TF3 const& rigid_omega,
            TF3 const& domain_min, TF3 const& domain_max, U3 const& resolution,
            TF3 const& cell_size, U num_nodes, TF sign) {
          runner.launch_compute_particle_boundary(
              distance, volume_grid, distance_grid, rigid_x, rigid_q,
              boundary_id, domain_min, domain_max, resolution, cell_size, sign,
              sample_x, sample_boundary, sample_boundary_kernel, num_samples);
        });
  }
  void transform_all_pellets() {
    pile.for_each_rigid(
        [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
            Variable<1, TF3> const& local_pellet_x, U pellet_index_offset,
            Variable<1, TF> const& distance_grid,
            Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
            TF3 const& rigid_v, TQ const& rigid_q, TF3 const& rigid_omega,
            TF3 const& domain_min, TF3 const& domain_max, U3 const& resolution,
            TF3 const& cell_size, U num_nodes, TF sign) {
          U num_rigid_particles = local_pellet_x.get_linear_shape();
          runner.launch(
              num_rigid_particles,
              [&](U grid_size, U block_size) {
                transform_pellets<<<grid_size, block_size>>>(
                    local_pellet_x, *particle_x, *particle_v, rigid_x, rigid_v,
                    rigid_q, rigid_omega, num_rigid_particles,
                    max_num_particles + pellet_index_offset);
              },
              "transform_pellets", transform_pellets<TQ, TF3>);
        });
  }
  template <U wrap>
  void update_particle_neighbors() {
    pid_length->set_zero();
    grid_anomaly->set_zero();

    if (num_particles == max_num_particles) {
      runner.launch_update_particle_grid(*particle_x, *pid, *pid_length,
                                         *grid_anomaly,
                                         num_particles + pile.num_pellets_);
    } else {
      runner.launch_update_particle_grid(*particle_x, *pid, *pid_length,
                                         *grid_anomaly, num_particles);
      runner.launch_update_particle_grid(*particle_x, *pid, *pid_length,
                                         *grid_anomaly, pile.num_pellets_,
                                         max_num_particles);
    }

    if (pile.volume_method_ == VolumeMethod::volume_map) {
      runner.template launch_make_neighbor_list<wrap>(
          *particle_x, *pid, *pid_length, *particle_neighbors,
          *particle_num_neighbors, *grid_anomaly, num_particles);
      runner.launch_compute_density(*particle_neighbors,
                                    *particle_num_neighbors, *particle_density,
                                    *particle_boundary_kernel, num_particles);
    } else {
      runner.template launch_make_bead_pellet_neighbor_list_check_contiguous<
          wrap>(*particle_x, *pid, *pid_length, *particle_neighbors,
                *particle_num_neighbors, *particle_boundary_neighbors,
                *particle_num_boundary_neighbors, *grid_anomaly,
                max_num_particles, num_particles, pile.num_pellets_);
      runner.launch_compute_density_with_pellets(
          *particle_neighbors, *particle_num_neighbors,
          *particle_boundary_neighbors, *particle_num_boundary_neighbors,
          *particle_density, num_particles);
    }
  }
  const U max_num_particles;
  U num_particles;
  TF t;
  TF dt;
  TF initial_dt;
  TF max_dt;
  TF min_dt;
  TF cfl;
  TF particle_radius;
  bool enable_surface_tension;
  bool enable_vorticity;
  TF next_emission_t;

  // report
  TF particle_max_v2;
  TF pile_max_v2;
  TF max_v2;
  TF cfl_dt;
  TF utilized_cfl;

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
  std::unique_ptr<Variable<2, TQ>> particle_boundary_neighbors;
  std::unique_ptr<Variable<1, U>> particle_num_boundary_neighbors;
  std::unique_ptr<Variable<1, U>> grid_anomaly;
};
}  // namespace alluvion

#endif
