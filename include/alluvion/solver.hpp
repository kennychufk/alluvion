#ifndef ALLUVION_SOLVER_HPP
#define ALLUVION_SOLVER_HPP

#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/usher.hpp"
namespace alluvion {
template <typename TF>
struct Solver {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float2, double2> TF2;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  Solver(TRunner& runner_arg, TPile& pile_arg, Store& store_arg,
         U max_num_particles_arg, U max_num_provisional_ghosts_arg = 0,
         U max_num_ghosts_arg = 0, U num_ushers = 0,
         bool enable_surface_tension_arg = false,
         bool enable_vorticity_arg = false, bool graphical = false)
      : max_num_provisional_ghosts(max_num_provisional_ghosts_arg),
        max_num_ghosts(max_num_ghosts_arg),
        max_num_particles(max_num_particles_arg),
        num_particles(0),
        num_provisional_ghosts(0),
        num_ghosts(0),
        store(store_arg),
        runner(runner_arg),
        pile(pile_arg),
        usher(new Usher<TF>(store_arg, pile_arg, num_ushers)),
        initial_dt(0),
        dt(0),
        t(0),
        particle_radius(store_arg.get_cn<TF>().particle_radius),
        ghost_boundary_volume_threshold(-1),
        ghost_fluid_density_threshold(store_arg.get_cn<TF>().density0 * 0.9),
        num_ghost_relaxation(5),
        relax_rate(0.05),
        next_emission_t(-1),
        particle_x(graphical ? store_arg.create_graphical<1, TF3>(
                                   {max_num_particles_arg + max_num_ghosts_arg})
                             : store_arg.create<1, TF3>({max_num_particles_arg +
                                                         max_num_ghosts_arg})),
        particle_v(store_arg.create<1, TF3>(
            {max_num_particles_arg + max_num_ghosts_arg})),
        particle_a(store_arg.create<1, TF3>({max_num_particles_arg})),
        particle_density(store_arg.create<1, TF>(
            {max_num_particles_arg + max_num_ghosts_arg})),
        particle_boundary(store_arg.create<2, TQ>(
            {pile.get_size(), max_num_particles_arg + max_num_ghosts_arg})),
        particle_boundary_kernel(store_arg.create<2, TQ>(
            {pile.get_size(), max_num_particles_arg + max_num_ghosts_arg})),
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
            {store_arg.get_cni().grid_res.x, store_arg.get_cni().grid_res.y,
             store_arg.get_cni().grid_res.z,
             store_arg.get_cni().max_num_particles_per_cell})),
        pid_length(store_arg.create<3, U>({store_arg.get_cni().grid_res.x,
                                           store_arg.get_cni().grid_res.y,
                                           store_arg.get_cni().grid_res.z})),
        pid_length_before_ghost(
            max_num_ghosts_arg > 0
                ? store_arg.create<3, U>({store_arg.get_cni().grid_res.x,
                                          store_arg.get_cni().grid_res.y,
                                          store_arg.get_cni().grid_res.z})
                : new Variable<3, U>()),
        particle_neighbors(store_arg.create<2, TQ>(
            {max_num_particles_arg + max_num_ghosts_arg,
             store_arg.get_cni().max_num_neighbors_per_particle})),
        particle_num_neighbors(store_arg.create<1, U>(
            {max_num_particles_arg + max_num_ghosts_arg})),
        particle_hcp_ipos(max_num_ghosts_arg > 0
                              ? store_arg.create<1, U2>({max_num_particles_arg})
                              : new Variable<1, U2>()),
        need_ghost(
            max_num_ghosts_arg > 0
                ? store_arg.create<3, U>({store_arg.get_cni().hcp_grid_res.x,
                                          store_arg.get_cni().hcp_grid_res.y,
                                          store_arg.get_cni().hcp_grid_res.z})
                : new Variable<3, U>()),
        provisional_ghost_boundary(
            max_num_ghosts_arg > 0
                ? store_arg.create<1, TQ>({max_num_provisional_ghosts_arg})
                : new Variable<1, TQ>()),
        particle_lagrange_multiplier(
            max_num_ghosts_arg > 0
                ? store_arg.create<1, TF>(
                      {max_num_particles_arg + max_num_ghosts_arg})
                : new Variable<1, TF>()),
        particle_density_before_ghost(
            max_num_ghosts_arg > 0
                ? store_arg.create<1, TF>({max_num_particles_arg})
                : new Variable<1, TF>()),
        enable_surface_tension(enable_surface_tension_arg),
        enable_vorticity(enable_vorticity_arg) {
    store.copy_cn<TF>();
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
    store.remove(*pid_length_before_ghost);
    store.remove(*particle_neighbors);
    store.remove(*particle_num_neighbors);
    store.remove(*particle_hcp_ipos);
    store.remove(*need_ghost);
    store.remove(*provisional_ghost_boundary);
    store.remove(*particle_lagrange_multiplier);
    store.remove(*particle_density_before_ghost);
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
  void compute_all_boundaries() {
    pile.for_each_rigid(
        [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
            Variable<1, TF> const& distance_grid,
            Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
            TQ const& rigid_q, TF3 const& domain_min, TF3 const& domain_max,
            U3 const& resolution, TF3 const& cell_size, U num_nodes, TF sign) {
          runner.launch_compute_particle_boundary(
              distance, volume_grid, distance_grid, rigid_x, rigid_q,
              boundary_id, domain_min, domain_max, resolution, cell_size,
              num_nodes, sign, dt, *particle_x, *particle_boundary,
              *particle_boundary_kernel, num_particles);
        });
  }
  void sample_all_boundaries(Variable<1, TF3>& sample_x,
                             Variable<2, TQ>& sample_boundary,
                             Variable<2, TQ>& sample_boundary_kernel,
                             U num_samples) {
    pile.for_each_rigid(
        [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
            Variable<1, TF> const& distance_grid,
            Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
            TQ const& rigid_q, TF3 const& domain_min, TF3 const& domain_max,
            U3 const& resolution, TF3 const& cell_size, U num_nodes, TF sign) {
          runner.launch_compute_particle_boundary(
              distance, volume_grid, distance_grid, rigid_x, rigid_q,
              boundary_id, domain_min, domain_max, resolution, cell_size,
              num_nodes, sign, dt, sample_x, sample_boundary,
              sample_boundary_kernel, num_samples);
        });
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
  template <U wrap>
  void relax_ghosts(bool revert_pid_length = true) {
    if (revert_pid_length) {
      pid_length->set_from(*pid_length_before_ghost);
      particle_density->set_from(*particle_density_before_ghost, num_particles);
    } else {
      pid_length_before_ghost->set_from(*pid_length);
      particle_density_before_ghost->set_from(*particle_density, num_particles);
    }
    runner.launch_update_particle_grid(*particle_x, *pid, *pid_length,
                                       num_ghosts, num_particles);
    runner.template launch_make_neighbor_list<wrap>(
        *particle_x, *pid, *pid_length, *particle_neighbors,
        *particle_num_neighbors, num_particles + num_ghosts);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_density_and_lagrange_multiplier_for_fluid<<<grid_size,
                                                              block_size>>>(
              *particle_neighbors, *particle_num_neighbors, *particle_density,
              *particle_lagrange_multiplier, num_particles);
        },
        "compute_density_and_lagrange_multiplier_for_fluid",
        compute_density_and_lagrange_multiplier_for_fluid<TQ, TF>);
    runner.launch(
        num_ghosts,
        [&](U grid_size, U block_size) {
          compute_density_and_lagrange_multiplier_for_ghosts<<<grid_size,
                                                               block_size>>>(
              *particle_neighbors, *particle_num_neighbors, *particle_density,
              *particle_lagrange_multiplier, num_ghosts, num_particles);
        },
        "compute_density_and_lagrange_multiplier_for_ghosts",
        compute_density_and_lagrange_multiplier_for_ghosts<TQ, TF>);
    runner.launch(
        num_ghosts,
        [&](U grid_size, U block_size) {
          shift_using_lagrange_multipliers<<<grid_size, block_size>>>(
              *particle_density, *particle_neighbors, *particle_num_neighbors,
              *particle_lagrange_multiplier, *particle_x, num_ghosts,
              num_particles, relax_rate);
        },
        "shift_using_lagrange_multipliers",
        shift_using_lagrange_multipliers<TQ, TF3, TF>);
  }
  template <U wrap>
  void initialize_ghosts() {
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          mark_deficient<<<grid_size, block_size>>>(
              *particle_x, *particle_density, *particle_hcp_ipos,
              num_particles);
        },
        "mark_deficient", mark_deficient<TF3, TF>);
    U num_deficient = runner.partition_y_equal(
        *particle_hcp_ipos, static_cast<U>(0), num_particles);

    need_ghost->set_same(static_cast<I>(kFMax<U>));
    runner.launch(
        num_deficient,
        [&](U grid_size, U block_size) {
          expand_deficient<<<grid_size, block_size>>>(
              *particle_hcp_ipos, *need_ghost, num_deficient);
        },
        "expand_deficient", expand_deficient<U>);
    U num_full = 0;
    if (num_deficient < num_particles) {
      num_full = runner.partition_y_equal(*particle_hcp_ipos, static_cast<U>(1),
                                          num_particles - num_deficient,
                                          num_deficient);
    }
    U num_occupied = num_deficient + num_full;
    runner.launch(
        num_occupied,
        [&](U grid_size, U block_size) {
          clear_cells_with_fluid_particles<<<grid_size, block_size>>>(
              *particle_hcp_ipos, *need_ghost, num_occupied);
        },
        "clear_cells_with_fluid_particles",
        clear_cells_with_fluid_particles<U>);

    num_provisional_ghosts = runner.partition_unequal(
        *need_ghost, kFMax<U>, need_ghost->get_linear_shape());
    if (num_provisional_ghosts > max_num_provisional_ghosts) {
      std::cerr << "num_provisional_ghosts: " << num_provisional_ghosts << " > "
                << max_num_provisional_ghosts << std::endl;
      num_provisional_ghosts = max_num_provisional_ghosts;
    }
    std::cout << "num_provisional_ghosts = " << num_provisional_ghosts
              << std::endl;
    runner.launch(
        num_provisional_ghosts,
        [&](U grid_size, U block_size) {
          initialize_provisional_ghosts<<<grid_size, block_size>>>(
              *provisional_ghost_boundary, *need_ghost, num_provisional_ghosts);
        },
        "initialize_provisional_ghosts", initialize_provisional_ghosts<TQ>);

    if (ghost_boundary_volume_threshold >= 0) {
      pile.for_each_rigid(
          [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
              Variable<1, TF> const& distance_grid,
              Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
              TQ const& rigid_q, TF3 const& domain_min, TF3 const& domain_max,
              U3 const& resolution, TF3 const& cell_size, U num_nodes,
              TF sign) {
            runner.launch(
                num_provisional_ghosts,
                [&](U grid_size, U block_size) {
                  accumulate_provisional_boundary_volume<wrap>
                      <<<grid_size, block_size>>>(
                          volume_grid, rigid_x, rigid_q, domain_min, domain_max,
                          resolution, cell_size, *provisional_ghost_boundary,
                          num_provisional_ghosts);
                },
                "accumulate_provisional_boundary_volume",
                accumulate_provisional_boundary_volume<wrap, TQ, TF3, TF>);
          });
      num_provisional_ghosts = runner.partition_w_less(
          *provisional_ghost_boundary, ghost_boundary_volume_threshold,
          num_provisional_ghosts);
      std::cout << "num_provisional_ghosts = " << num_provisional_ghosts
                << std::endl;
    }

    runner.launch(
        num_provisional_ghosts,
        [&](U grid_size, U block_size) {
          compute_provisional_fluid_density<wrap>
              <<<grid_size, block_size>>>(*provisional_ghost_boundary, *pid,
                                          *pid_length, num_provisional_ghosts);
        },
        "compute_provisional_fluid_density",
        compute_provisional_fluid_density<wrap, TQ>);

    num_ghosts = runner.partition_w_less(*provisional_ghost_boundary,
                                         ghost_fluid_density_threshold,
                                         num_provisional_ghosts);

    if (num_ghosts > max_num_ghosts) {
      std::cerr << "num_ghosts: " << num_ghosts << " > " << max_num_ghosts
                << std::endl;
      num_ghosts = max_num_ghosts;
    }
    std::cout << "num_ghosts = " << num_ghosts << std::endl;

    runner.launch(
        num_ghosts,
        [&](U grid_size, U block_size) {
          append_ghosts<<<grid_size, block_size>>>(*provisional_ghost_boundary,
                                                   *particle_x, num_ghosts,
                                                   num_particles);
        },
        "append_ghosts", append_ghosts<TQ, TF3>);
  }
  template <U wrap>
  void prepare_ghosts() {
    if (max_num_ghosts > 0) {
      initialize_ghosts<wrap>();
      for (U relaxation_i = 0; relaxation_i < num_ghost_relaxation;
           ++relaxation_i) {
        relax_ghosts<wrap>(relaxation_i > 0);
      }
      pid_length->set_from(*pid_length_before_ghost);
      particle_density->set_from(*particle_density_before_ghost, num_particles);
      runner.launch_update_particle_grid(*particle_x, *pid, *pid_length,
                                         num_ghosts, num_particles);
      runner.template launch_make_neighbor_list<wrap>(
          *particle_x, *pid, *pid_length, *particle_neighbors,
          *particle_num_neighbors, num_particles + num_ghosts);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_density_and_lagrange_multiplier_for_fluid<<<grid_size,
                                                                block_size>>>(
                *particle_neighbors, *particle_num_neighbors, *particle_density,
                *particle_lagrange_multiplier, num_particles);
          },
          "compute_density_and_lagrange_multiplier_for_fluid",
          compute_density_and_lagrange_multiplier_for_fluid<TQ, TF>);
    }
  }

  U max_num_particles;
  U max_num_provisional_ghosts;
  U max_num_ghosts;
  U num_particles;
  U num_provisional_ghosts;
  U num_ghosts;
  TF t;
  TF dt;
  TF initial_dt;
  TF max_dt;
  TF min_dt;
  TF cfl;
  TF particle_radius;
  bool enable_surface_tension;
  bool enable_vorticity;
  TF ghost_boundary_volume_threshold;
  TF ghost_fluid_density_threshold;
  U num_ghost_relaxation;
  TF relax_rate;

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
  std::unique_ptr<Variable<3, U>> pid_length_before_ghost;
  std::unique_ptr<Variable<2, TQ>> particle_neighbors;
  std::unique_ptr<Variable<1, U>> particle_num_neighbors;

  std::unique_ptr<Variable<1, U2>> particle_hcp_ipos;
  std::unique_ptr<Variable<3, U>> need_ghost;
  std::unique_ptr<Variable<1, TQ>> provisional_ghost_boundary;
  std::unique_ptr<Variable<1, TF>> particle_lagrange_multiplier;
  std::unique_ptr<Variable<1, TF>> particle_density_before_ghost;
};
}  // namespace alluvion

#endif
