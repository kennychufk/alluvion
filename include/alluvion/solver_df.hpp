#ifndef ALLUVION_SOLVER_DF_HPP
#define ALLUVION_SOLVER_DF_HPP

#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"
namespace alluvion {
template <typename TF3, typename TQ, typename TF>
struct SolverDf {
  using TPile = Pile<TF3, TQ, TF>;
  SolverDf(Runner& runner_arg, TPile& pile_arg,
           Variable<1, TF3>& particle_x_arg,
           Variable<1, TF>& particle_normalized_attr_arg,
           Variable<1, TF3>& particle_v_arg, Variable<1, TF3>& particle_a_arg,
           Variable<1, TF>& particle_density_arg,
           Variable<2, TF3>& particle_boundary_xj_arg,
           Variable<2, TF>& particle_boundary_volume_arg,
           Variable<2, TF3>& particle_force_arg,
           Variable<2, TF3>& particle_torque_arg,
           Variable<1, TF>& particle_cfl_v2_arg,

           Variable<1, TF>& particle_dfsph_factor_arg,
           Variable<1, TF>& particle_kappa_arg,
           Variable<1, TF>& particle_kappa_v_arg,
           Variable<1, TF>& particle_density_adv_arg,

           Variable<4, Q>& pid_arg, Variable<3, U>& pid_length_arg,
           Variable<2, Q>& particle_neighbors_arg,
           Variable<1, U>& particle_num_neighbors_arg)
      : runner(runner_arg),
        pile(pile_arg),
        particle_x(particle_x_arg),
        particle_normalized_attr(particle_normalized_attr_arg),
        particle_v(particle_v_arg),
        particle_a(particle_a_arg),
        particle_density(particle_density_arg),
        particle_boundary_xj(particle_boundary_xj_arg),
        particle_boundary_volume(particle_boundary_volume_arg),
        particle_force(particle_force_arg),
        particle_torque(particle_torque_arg),
        particle_cfl_v2(particle_cfl_v2_arg),

        particle_dfsph_factor(particle_dfsph_factor_arg),
        particle_kappa(particle_kappa_arg),
        particle_kappa_v(particle_kappa_v_arg),
        particle_density_adv(particle_density_adv_arg),

        pid(pid_arg),
        pid_length(pid_length_arg),
        particle_neighbors(particle_neighbors_arg),
        particle_num_neighbors(particle_num_neighbors_arg) {}

  template <U wrap>
  void step() {
    particle_force.set_zero();
    particle_torque.set_zero();
    pile.copy_kinematics_to_device();
    pile.for_each_rigid([&](U boundary_id, Variable<1, TF> const& distance_grid,
                            Variable<1, TF> const& volume_grid,
                            TF3 const& rigid_x, Q const& rigid_q,
                            TF3 const& domain_min, TF3 const& domain_max,
                            U3 const& resolution, TF3 const& cell_size,
                            U num_nodes, TF sign, TF thickness) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary<wrap><<<grid_size, block_size>>>(
                volume_grid, distance_grid, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, num_nodes, 0,
                sign, thickness, dt, particle_x, particle_v,
                particle_boundary_xj, particle_boundary_volume, num_particles);
          },
          "compute_particle_boundary",
          compute_particle_boundary<wrap, TQ, TF3, TF>);
    });
    pid_length.set_zero();
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          update_particle_grid<<<grid_size, block_size>>>(
              particle_x, pid, pid_length, num_particles);
        },
        "update_particle_grid", update_particle_grid<TQ, TF3>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          make_neighbor_list<wrap><<<grid_size, block_size>>>(
              particle_x, pid, pid_length, particle_neighbors,
              particle_num_neighbors, num_particles);
        },
        "make_neighbor_list", make_neighbor_list<wrap, TQ, TF3>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_density<<<grid_size, block_size>>>(
              particle_x, particle_neighbors, particle_num_neighbors,
              particle_density, particle_boundary_xj, particle_boundary_volume,
              num_particles);
        },
        "compute_density", compute_density<TQ, TF3, TF>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_dfsph_factor<<<grid_size, block_size>>>(
              particle_x, particle_neighbors, particle_num_neighbors,
              particle_dfsph_factor, particle_boundary_xj,
              particle_boundary_volume, num_particles);
        },
        "compute_dfsph_factor", compute_dfsph_factor<TQ, TF3, TF>);
    // ===== [divergence solve
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          warm_start_divergence_solve_0<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_kappa_v, particle_neighbors,
              particle_num_neighbors, particle_boundary_xj,
              particle_boundary_volume, pile.x_device_, pile.v_device_,
              pile.omega_device_, dt, num_particles);
        },
        "warm_start_divergence_solve_0",
        warm_start_divergence_solve_0<TQ, TF3, TF>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          warm_start_divergence_solve_1<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_kappa_v, particle_neighbors,
              particle_num_neighbors, particle_boundary_xj,
              particle_boundary_volume, particle_force, particle_torque,
              pile.x_device_, pile.v_device_, pile.omega_device_, dt,
              num_particles);
        },
        "warm_start_divergence_solve_1",
        warm_start_divergence_solve_1<TQ, TF3, TF>);

    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_velocity_of_density_change<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_dfsph_factor,
              particle_density_adv, particle_neighbors, particle_num_neighbors,
              particle_boundary_xj, particle_boundary_volume, pile.x_device_,
              pile.v_device_, pile.omega_device_, dt, num_particles);
        },
        "compute_velocity_of_density_change",
        compute_velocity_of_density_change<TQ, TF3, TF>);

    TF avg_density_err = std::numeric_limits<TF>::max();
    constexpr TF kMaxError = 1e-3;
    U num_divergence_solve_iteration = 0;
    while (num_divergence_solve_iteration < 1 ||
           avg_density_err > kMaxError / dt) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            divergence_solve_iteration<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_dfsph_factor,
                particle_density_adv, particle_kappa_v, particle_neighbors,
                particle_num_neighbors, particle_boundary_xj,
                particle_boundary_volume, particle_force, particle_torque,
                pile.x_device_, pile.v_device_, pile.omega_device_, dt,
                num_particles);
          },
          "divergence_solve_iteration",
          divergence_solve_iteration<TQ, TF3, TF>);

      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_divergence_solve_density_error<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_density_adv,
                particle_neighbors, particle_num_neighbors,
                particle_boundary_xj, particle_boundary_volume, pile.x_device_,
                pile.v_device_, pile.omega_device_, dt, num_particles);
          },
          "compute_divergence_solve_density_error",
          compute_divergence_solve_density_error<TQ, TF3, TF>);
      avg_density_err =
          Runner::sum(particle_density_adv, num_particles) / num_particles;
      ++num_divergence_solve_iteration;
    }
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          divergence_solve_finish<<<grid_size, block_size>>>(
              particle_dfsph_factor, particle_kappa_v, dt, num_particles);
        },
        "divergence_solve_finish", divergence_solve_finish<TF>);
    // ===== ]divergence solve

    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          clear_acceleration<<<grid_size, block_size>>>(particle_a,
                                                        num_particles);
        },
        "clear_acceleration", clear_acceleration<TF3>);
    // compute_normal
    // compute_surface_tension_fluid
    // compute_surface_tension_boundary

    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_viscosity<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_density, particle_neighbors,
              particle_num_neighbors, particle_a, particle_force,
              particle_torque, particle_boundary_xj, particle_boundary_volume,
              pile.x_device_, pile.v_device_, pile.omega_device_,
              pile.boundary_viscosity_device_, num_particles);
        },
        "compute_viscosity", compute_viscosity<TQ, TF3, TF>);

    // reset_angular_acceleration
    // compute_vorticity_fluid
    // compute_vorticity_boundary
    // integrate_angular_acceleration
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          calculate_cfl_v2<<<grid_size, block_size>>>(
              particle_v, particle_a, particle_cfl_v2, dt, num_particles);
        },
        "calculate_cfl_v2", calculate_cfl_v2<TF3, TF>);
    TF particle_max_v2 = Runner::max(particle_cfl_v2, num_particles);
    TF pile_max_v2 = pile.calculate_cfl_v2();
    TF max_v2 = max(particle_max_v2, pile_max_v2);
    dt = cfl * ((particle_radius * 2) / sqrt(max_v2));
    dt = max(min(dt, max_dt), min_dt);

    // update_dt

    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          integrate_non_pressure_acceleration<<<grid_size, block_size>>>(
              particle_v, particle_a, dt, num_particles);
        },
        "integrate_non_pressure_acceleration",
        integrate_non_pressure_acceleration<TF3, TF>);
    // ===== [pressure solve
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          warm_start_pressure_solve0<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_density, particle_kappa,
              particle_neighbors, particle_num_neighbors, particle_boundary_xj,
              particle_boundary_volume, pile.x_device_, pile.v_device_,
              pile.omega_device_, dt, num_particles);
        },
        "warm_start_pressure_solve0", warm_start_pressure_solve0<TQ, TF3, TF>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          warm_start_pressure_solve1<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_kappa, particle_neighbors,
              particle_num_neighbors, particle_boundary_xj,
              particle_boundary_volume, particle_force, particle_torque,
              pile.x_device_, pile.v_device_, pile.omega_device_, dt,
              num_particles);
        },
        "warm_start_pressure_solve1", warm_start_pressure_solve1<TQ, TF3, TF>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_rho_adv<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_density, particle_dfsph_factor,
              particle_density_adv, particle_neighbors, particle_num_neighbors,
              particle_boundary_xj, particle_boundary_volume, pile.x_device_,
              pile.v_device_, pile.omega_device_, dt, num_particles);
        },
        "compute_rho_adv", compute_rho_adv<TQ, TF3, TF>);
    avg_density_err = std::numeric_limits<TF>::max();
    U num_pressure_solve_iteration = 0;
    while (num_pressure_solve_iteration < 2 || avg_density_err > kMaxError) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            pressure_solve_iteration<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_dfsph_factor,
                particle_density_adv, particle_kappa, particle_neighbors,
                particle_num_neighbors, particle_boundary_xj,
                particle_boundary_volume, particle_force, particle_torque,
                pile.x_device_, pile.v_device_, pile.omega_device_, dt,
                num_particles);
          },
          "pressure_solve_iteration", pressure_solve_iteration<TQ, TF3, TF>);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_pressure_solve_density_error<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_density, particle_density_adv,
                particle_neighbors, particle_num_neighbors,
                particle_boundary_xj, particle_boundary_volume, pile.x_device_,
                pile.v_device_, pile.omega_device_, dt, num_particles);
          },
          "compute_pressure_solve_density_error",
          compute_pressure_solve_density_error<TQ, TF3, TF>);
      avg_density_err =
          Runner::sum(particle_density_adv, num_particles) / num_particles -
          1._F;  // TODO: add variable to store particle_density_adv - 1
                 // (better numerical precision)
      ++num_pressure_solve_iteration;
    }
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          pressure_solve_finish<<<grid_size, block_size>>>(particle_kappa, dt,
                                                           num_particles);
        },
        "pressure_solve_finish", pressure_solve_finish<TF>);
    // ===== ]pressure solve
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          integrate_velocity<wrap><<<grid_size, block_size>>>(
              particle_x, particle_v, dt, num_particles);
        },
        "integrate_velocity", integrate_velocity<wrap, TF3, TF>);

    // rigids
    for (U i = 0; i < pile.get_size(); ++i) {
      if (pile.mass_[i] == 0._F) continue;
      pile.force_[i] =
          Runner::sum(particle_force, num_particles, i * num_particles);
      pile.torque_[i] =
          Runner::sum(particle_torque, num_particles, i * num_particles);
    }

    pile.integrate_kinematics(dt);
    pile.find_contacts();
    pile.solve_contacts();
  }
  void colorize(TF lower_bound, TF upper_bound) {
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          scale<<<grid_size, block_size>>>(
              particle_kappa_v, particle_normalized_attr, lower_bound,
              upper_bound, num_particles);
        },
        "scale", scale<TF>);

    // runner.launch(
    //     num_particles,
    //     [&](U grid_size, U block_size) {
    //       normalize_vector_magnitude<<<grid_size, block_size>>>(
    //           particle_v, particle_normalized_attr, lower_bound, upper_bound,
    //           num_particles);
    //     },
    //     "normalize_vector_magnitude", normalize_vector_magnitude<TF3, TF>);
  }
  Runner& runner;
  TPile& pile;

  Variable<1, TF3>& particle_x;
  Variable<1, TF>& particle_normalized_attr;
  Variable<1, TF3>& particle_v;
  Variable<1, TF3>& particle_a;
  Variable<1, TF>& particle_density;
  Variable<2, TF3>& particle_boundary_xj;
  Variable<2, TF>& particle_boundary_volume;
  Variable<2, TF3>& particle_force;
  Variable<2, TF3>& particle_torque;
  Variable<1, TF>& particle_cfl_v2;

  Variable<1, TF>& particle_dfsph_factor;
  Variable<1, TF>& particle_kappa;
  Variable<1, TF>& particle_kappa_v;
  Variable<1, TF>& particle_density_adv;

  Variable<4, TQ>& pid;
  Variable<3, U>& pid_length;
  Variable<2, TQ>& particle_neighbors;
  Variable<1, U>& particle_num_neighbors;

  U num_particles;
  TF dt;
  TF max_dt;
  TF min_dt;
  TF cfl;
  TF particle_radius;
};
}  // namespace alluvion

#endif /* ALLUVION_SOLVER_DF_HPP */
