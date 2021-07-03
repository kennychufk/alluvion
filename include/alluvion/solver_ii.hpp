#ifndef ALLUVION_SOLVER_II_HPP
#define ALLUVION_SOLVER_II_HPP

#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/solver.hpp"
#include "alluvion/store.hpp"
namespace alluvion {
template <typename TF>
struct SolverIi : public Solver<TF> {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  using Base = Solver<TF>;
  SolverIi(TRunner& runner_arg, TPile& pile_arg,
           Variable<1, TF3>& particle_x_arg,
           Variable<1, TF>& particle_normalized_attr_arg,
           Variable<1, TF3>& particle_v_arg, Variable<1, TF3>& particle_a_arg,
           Variable<1, TF>& particle_density_arg,
           Variable<2, TF3>& particle_boundary_xj_arg,
           Variable<2, TF>& particle_boundary_volume_arg,
           Variable<2, TF3>& particle_force_arg,
           Variable<2, TF3>& particle_torque_arg,
           Variable<1, TF>& particle_cfl_v2_arg,

           Variable<1, TF>& particle_pressure_arg,
           Variable<1, TF>& particle_last_pressure_arg,
           Variable<1, TF>& particle_aii_arg,
           Variable<1, TF3>& particle_dii_arg,
           Variable<1, TF3>& particle_dij_pj_arg,
           Variable<1, TF>& particle_sum_tmp_arg,
           Variable<1, TF>& particle_adv_density_arg,
           Variable<1, TF3>& particle_pressure_accel_arg,
           Variable<1, TF>& particle_density_err_arg,

           Variable<4, TQ>& pid_arg, Variable<3, U>& pid_length_arg,
           Variable<2, TQ>& particle_neighbors_arg,
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

        particle_pressure(particle_pressure_arg),
        particle_last_pressure(particle_last_pressure_arg),
        particle_aii(particle_aii_arg),
        particle_dii(particle_dii_arg),
        particle_dij_pj(particle_dij_pj_arg),
        particle_sum_tmp(particle_sum_tmp_arg),
        particle_adv_density(particle_adv_density_arg),
        particle_pressure_accel(particle_pressure_accel_arg),
        particle_density_err(particle_density_err_arg),

        pid(pid_arg),
        pid_length(pid_length_arg),
        particle_neighbors(particle_neighbors_arg),
        particle_num_neighbors(particle_num_neighbors_arg) {}

  template <U wrap>
  void step() {
    particle_force.set_zero();
    particle_torque.set_zero();
    pile.copy_kinematics_to_device();
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          clear_acceleration<<<grid_size, block_size>>>(particle_a,
                                                        Base::num_particles);
        },
        "clear_acceleration", clear_acceleration<TF3>);
    pile.for_each_rigid([&](U boundary_id, Variable<1, TF> const& distance_grid,
                            Variable<1, TF> const& volume_grid,
                            TF3 const& rigid_x, TQ const& rigid_q,
                            TF3 const& domain_min, TF3 const& domain_max,
                            U3 const& resolution, TF3 const& cell_size,
                            U num_nodes, TF sign, TF thickness) {
      runner.launch(
          Base::num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary<wrap><<<grid_size, block_size>>>(
                volume_grid, distance_grid, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, num_nodes, 0,
                sign, thickness, Base::dt, particle_x, particle_v,
                particle_boundary_xj, particle_boundary_volume,
                Base::num_particles);
          },
          "compute_particle_boundary",
          compute_particle_boundary<wrap, TQ, TF3, TF>);
    });
    pid_length.set_zero();
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          update_particle_grid<<<grid_size, block_size>>>(
              particle_x, pid, pid_length, Base::num_particles);
        },
        "update_particle_grid", update_particle_grid<TQ, TF3>);
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          make_neighbor_list<wrap><<<grid_size, block_size>>>(
              particle_x, pid, pid_length, particle_neighbors,
              particle_num_neighbors, Base::num_particles);
        },
        "make_neighbor_list", make_neighbor_list<wrap, TQ, TF3>);
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          compute_density<<<grid_size, block_size>>>(
              particle_x, particle_neighbors, particle_num_neighbors,
              particle_density, particle_boundary_xj, particle_boundary_volume,
              Base::num_particles);
        },
        "compute_density", compute_density<TQ, TF3, TF>);
    // compute_normal
    // compute_surface_tension_fluid
    // compute_surface_tension_boundary

    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          compute_viscosity<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_density, particle_neighbors,
              particle_num_neighbors, particle_a, particle_force,
              particle_torque, particle_boundary_xj, particle_boundary_volume,
              pile.x_device_, pile.v_device_, pile.omega_device_,
              Base::num_particles);
        },
        "compute_viscosity", compute_viscosity<TQ, TF3, TF>);

    // reset_angular_acceleration
    // compute_vorticity_fluid
    // compute_vorticity_boundary
    // integrate_angular_acceleration
    //
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          calculate_cfl_v2<<<grid_size, block_size>>>(particle_v, particle_a,
                                                      particle_cfl_v2, Base::dt,
                                                      Base::num_particles);
        },
        "calculate_cfl_v2", calculate_cfl_v2<TF3, TF>);
    TF particle_max_v2 = TRunner::max(particle_cfl_v2, Base::num_particles);
    TF pile_max_v2 = pile.calculate_cfl_v2();
    TF max_v2 = max(particle_max_v2, pile_max_v2);
    Base::dt = Base::cfl * ((Base::particle_radius * 2) / sqrt(max_v2));
    Base::dt = max(min(Base::dt, Base::max_dt), Base::min_dt);
    // update_dt

    // ===== [solve
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          predict_advection0<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_a, particle_density,
              particle_dii, particle_pressure, particle_last_pressure,
              particle_neighbors, particle_num_neighbors, particle_boundary_xj,
              particle_boundary_volume, Base::dt, Base::num_particles);
        },
        "predict_advection0", predict_advection0<TQ, TF3, TF>);
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          predict_advection1<<<grid_size, block_size>>>(
              particle_x, particle_v, particle_dii, particle_adv_density,
              particle_aii, particle_density, particle_neighbors,
              particle_num_neighbors, particle_boundary_xj,
              particle_boundary_volume, pile.x_device_, pile.v_device_,
              pile.omega_device_, Base::dt, Base::num_particles);
        },
        "predict_advection1", predict_advection1<TQ, TF3, TF>);
    TF avg_density_err = std::numeric_limits<TF>::max();
    constexpr TF kMaxError = 1e-3;
    U num_solve_iteration = 0;
    while (num_solve_iteration < 2 || avg_density_err > kMaxError) {
      runner.launch(
          Base::num_particles,
          [&](U grid_size, U block_size) {
            pressure_solve_iteration0<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_last_pressure,
                particle_dij_pj, particle_neighbors, particle_num_neighbors,
                Base::num_particles);
          },
          "pressure_solve_iteration0", pressure_solve_iteration0<TQ, TF3, TF>);
      runner.launch(
          Base::num_particles,
          [&](U grid_size, U block_size) {
            pressure_solve_iteration1<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_last_pressure,
                particle_dii, particle_dij_pj, particle_sum_tmp,
                particle_neighbors, particle_num_neighbors,
                particle_boundary_xj, particle_boundary_volume,
                Base::num_particles);
          },
          "pressure_solve_iteration1", pressure_solve_iteration1<TQ, TF3, TF>);
      runner.launch(
          Base::num_particles,
          [&](U grid_size, U block_size) {
            pressure_solve_iteration1_summarize<<<grid_size, block_size>>>(
                particle_aii, particle_adv_density, particle_sum_tmp,
                particle_last_pressure, particle_pressure, particle_density_err,
                Base::dt, Base::num_particles);
          },
          "pressure_solve_iteration1_summarize",
          pressure_solve_iteration1_summarize<TF>);
      avg_density_err =
          TRunner::sum(particle_density_err, Base::num_particles) /
          Base::num_particles;
      ++num_solve_iteration;
    }
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          compute_pressure_accels<<<grid_size, block_size>>>(
              particle_x, particle_density, particle_pressure,
              particle_pressure_accel, particle_neighbors,
              particle_num_neighbors, particle_force, particle_torque,
              particle_boundary_xj, particle_boundary_volume, pile.x_device_,
              Base::num_particles);
        },
        "compute_pressure_accels", compute_pressure_accels<TQ, TF3, TF>);
    // ===== ]divergence solve
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          kinematic_integration<wrap><<<grid_size, block_size>>>(
              particle_x, particle_v, particle_pressure_accel, Base::dt,
              Base::num_particles);
        },
        "kinematic_integration", kinematic_integration<wrap, TF3, TF>);

    // rigids
    for (U i = 0; i < pile.get_size(); ++i) {
      if (pile.mass_[i] == 0) continue;
      pile.force_[i] = TRunner::sum(particle_force, Base::num_particles,
                                    i * Base::num_particles);
      pile.torque_[i] = TRunner::sum(particle_torque, Base::num_particles,
                                     i * Base::num_particles);
    }

    pile.integrate_kinematics(Base::dt);
    pile.find_contacts();
    pile.solve_contacts();
  }

  void colorize_speed(TF lower_bound, TF upper_bound) {
    runner.launch(
        Base::num_particles,
        [&](U grid_size, U block_size) {
          normalize_vector_magnitude<<<grid_size, block_size>>>(
              particle_v, particle_normalized_attr, lower_bound, upper_bound,
              Base::num_particles);
        },
        "normalize_vector_magnitude", normalize_vector_magnitude<TF3, TF>);
  }
  TRunner& runner;
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

  Variable<1, TF>& particle_pressure;
  Variable<1, TF>& particle_last_pressure;
  Variable<1, TF>& particle_aii;
  Variable<1, TF3>& particle_dii;
  Variable<1, TF3>& particle_dij_pj;
  Variable<1, TF>& particle_sum_tmp;
  Variable<1, TF>& particle_adv_density;
  Variable<1, TF3>& particle_pressure_accel;
  Variable<1, TF>& particle_density_err;

  Variable<4, TQ>& pid;
  Variable<3, U>& pid_length;
  Variable<2, TQ>& particle_neighbors;
  Variable<1, U>& particle_num_neighbors;
};
}  // namespace alluvion

#endif /* ALLUVION_SOLVER_II_HPP */
