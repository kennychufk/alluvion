#ifndef ALLUVION_SOLVER_II_HPP
#define ALLUVION_SOLVER_II_HPP

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
  using Base::cfl;
  using Base::dt;
  using Base::max_dt;
  using Base::min_dt;
  using Base::num_particles;
  using Base::particle_radius;
  using Base::pile;
  using Base::runner;
  using Base::t;
  SolverIi(TRunner& runner_arg, TPile& pile_arg, Store& store,
           U max_num_particles, U3 grid_res, U max_num_particles_per_cell = 64,
           U max_num_neighbors_per_particle = 64, bool graphical = false)
      : Base(runner_arg, pile_arg),
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

        particle_pressure(store.create<1, TF>({max_num_particles})),
        particle_last_pressure(store.create<1, TF>({max_num_particles})),
        particle_aii(store.create<1, TF>({max_num_particles})),
        particle_dii(store.create<1, TF3>({max_num_particles})),
        particle_dij_pj(store.create<1, TF3>({max_num_particles})),
        particle_sum_tmp(store.create<1, TF>({max_num_particles})),
        particle_adv_density(store.create<1, TF>({max_num_particles})),
        particle_pressure_accel(store.create<1, TF3>({max_num_particles})),
        particle_density_err(store.create<1, TF>({max_num_particles})),

        pid(store.create<4, TQ>(
            {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell})),
        pid_length(store.create<3, U>({grid_res.x, grid_res.y, grid_res.z})),
        particle_neighbors(store.create<2, TQ>(
            {max_num_particles, max_num_neighbors_per_particle})),
        particle_num_neighbors(store.create<1, U>({max_num_particles})) {}

  template <U wrap, U gravitation>
  void step() {
    particle_force->set_zero();
    particle_torque->set_zero();
    pile.copy_kinematics_to_device();
    if constexpr (gravitation == 0) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            clear_acceleration<<<grid_size, block_size>>>(*particle_a,
                                                          num_particles);
          },
          "clear_acceleration", clear_acceleration<TF3>);
    } else if constexpr (gravitation == 1) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            apply_axial_gravitation<<<grid_size, block_size>>>(
                *particle_a, *particle_x, num_particles);
          },
          "apply_axial_gravitation", apply_axial_gravitation<TF3>);
    }
    pile.for_each_rigid([&](U boundary_id, Variable<1, TF> const& distance_grid,
                            Variable<1, TF> const& volume_grid,
                            TF3 const& rigid_x, TQ const& rigid_q,
                            TF3 const& domain_min, TF3 const& domain_max,
                            U3 const& resolution, TF3 const& cell_size,
                            U num_nodes, TF sign, TF thickness) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary<wrap><<<grid_size, block_size>>>(
                volume_grid, distance_grid, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, num_nodes, 0,
                sign, thickness, dt, *particle_x, *particle_v,
                *particle_boundary_xj, *particle_boundary_volume,
                num_particles);
          },
          "compute_particle_boundary",
          compute_particle_boundary<wrap, TQ, TF3, TF>);
    });
    pid_length->set_zero();
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          update_particle_grid<<<grid_size, block_size>>>(
              *particle_x, *pid, *pid_length, num_particles);
        },
        "update_particle_grid", update_particle_grid<TQ, TF3>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          make_neighbor_list<wrap><<<grid_size, block_size>>>(
              *particle_x, *pid, *pid_length, *particle_neighbors,
              *particle_num_neighbors, num_particles);
        },
        "make_neighbor_list", make_neighbor_list<wrap, TQ, TF3>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_density<<<grid_size, block_size>>>(
              *particle_x, *particle_neighbors, *particle_num_neighbors,
              *particle_density, *particle_boundary_xj,
              *particle_boundary_volume, num_particles);
        },
        "compute_density", compute_density<TQ, TF3, TF>);
    // compute_normal
    // compute_surface_tension_fluid
    // compute_surface_tension_boundary

    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_viscosity<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_density, *particle_neighbors,
              *particle_num_neighbors, *particle_a, *particle_force,
              *particle_torque, *particle_boundary_xj,
              *particle_boundary_volume, *pile.x_device_, *pile.v_device_,
              *pile.omega_device_, num_particles);
        },
        "compute_viscosity", compute_viscosity<TQ, TF3, TF>);

    // reset_angular_acceleration
    // compute_vorticity_fluid
    // compute_vorticity_boundary
    // integrate_angular_acceleration
    //
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          calculate_cfl_v2<<<grid_size, block_size>>>(
              *particle_v, *particle_a, *particle_cfl_v2, dt, num_particles);
        },
        "calculate_cfl_v2", calculate_cfl_v2<TF3, TF>);
    TF particle_max_v2 = TRunner::max(*particle_cfl_v2, num_particles);
    TF pile_max_v2 = pile.calculate_cfl_v2();
    TF max_v2 = max(particle_max_v2, pile_max_v2);
    dt = cfl * ((particle_radius * 2) / sqrt(max_v2));
    dt = max(min(dt, max_dt), min_dt);
    // update_dt

    // ===== [solve
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          predict_advection0<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_a, *particle_density,
              *particle_dii, *particle_pressure, *particle_last_pressure,
              *particle_neighbors, *particle_num_neighbors,
              *particle_boundary_xj, *particle_boundary_volume, dt,
              num_particles);
        },
        "predict_advection0", predict_advection0<TQ, TF3, TF>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          predict_advection1<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_dii, *particle_adv_density,
              *particle_aii, *particle_density, *particle_neighbors,
              *particle_num_neighbors, *particle_boundary_xj,
              *particle_boundary_volume, *pile.x_device_, *pile.v_device_,
              *pile.omega_device_, dt, num_particles);
        },
        "predict_advection1", predict_advection1<TQ, TF3, TF>);
    TF avg_density_err = std::numeric_limits<TF>::max();
    constexpr TF kMaxError = 1e-3;
    U num_solve_iteration = 0;
    while (num_solve_iteration < 2 || avg_density_err > kMaxError) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            pressure_solve_iteration0<<<grid_size, block_size>>>(
                *particle_x, *particle_density, *particle_last_pressure,
                *particle_dij_pj, *particle_neighbors, *particle_num_neighbors,
                num_particles);
          },
          "pressure_solve_iteration0", pressure_solve_iteration0<TQ, TF3, TF>);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            pressure_solve_iteration1<<<grid_size, block_size>>>(
                *particle_x, *particle_density, *particle_last_pressure,
                *particle_dii, *particle_dij_pj, *particle_sum_tmp,
                *particle_neighbors, *particle_num_neighbors,
                *particle_boundary_xj, *particle_boundary_volume,
                num_particles);
          },
          "pressure_solve_iteration1", pressure_solve_iteration1<TQ, TF3, TF>);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            pressure_solve_iteration1_summarize<<<grid_size, block_size>>>(
                *particle_aii, *particle_adv_density, *particle_sum_tmp,
                *particle_last_pressure, *particle_pressure,
                *particle_density_err, dt, num_particles);
          },
          "pressure_solve_iteration1_summarize",
          pressure_solve_iteration1_summarize<TF>);
      avg_density_err =
          TRunner::sum(*particle_density_err, num_particles) / num_particles;
      ++num_solve_iteration;
    }
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_pressure_accels<<<grid_size, block_size>>>(
              *particle_x, *particle_density, *particle_pressure,
              *particle_pressure_accel, *particle_neighbors,
              *particle_num_neighbors, *particle_force, *particle_torque,
              *particle_boundary_xj, *particle_boundary_volume, *pile.x_device_,
              num_particles);
        },
        "compute_pressure_accels", compute_pressure_accels<TQ, TF3, TF>);
    // ===== ]divergence solve
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          kinematic_integration<wrap><<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_pressure_accel, dt,
              num_particles);
        },
        "kinematic_integration", kinematic_integration<wrap, TF3, TF>);

    // rigids
    for (U i = 0; i < pile.get_size(); ++i) {
      if (pile.mass_[i] == 0) continue;
      pile.force_[i] =
          TRunner::sum(*particle_force, num_particles, i * num_particles);
      pile.torque_[i] =
          TRunner::sum(*particle_torque, num_particles, i * num_particles);
    }

    pile.integrate_kinematics(dt);
    pile.find_contacts();
    pile.solve_contacts();
    t += dt;
  }

  std::unique_ptr<Variable<1, TF3>> particle_x;
  std::unique_ptr<Variable<1, TF3>> particle_v;
  std::unique_ptr<Variable<1, TF3>> particle_a;
  std::unique_ptr<Variable<1, TF>> particle_density;
  std::unique_ptr<Variable<2, TF3>> particle_boundary_xj;
  std::unique_ptr<Variable<2, TF>> particle_boundary_volume;
  std::unique_ptr<Variable<2, TF3>> particle_force;
  std::unique_ptr<Variable<2, TF3>> particle_torque;
  std::unique_ptr<Variable<1, TF>> particle_cfl_v2;

  std::unique_ptr<Variable<1, TF>> particle_pressure;
  std::unique_ptr<Variable<1, TF>> particle_last_pressure;
  std::unique_ptr<Variable<1, TF>> particle_aii;
  std::unique_ptr<Variable<1, TF3>> particle_dii;
  std::unique_ptr<Variable<1, TF3>> particle_dij_pj;
  std::unique_ptr<Variable<1, TF>> particle_sum_tmp;
  std::unique_ptr<Variable<1, TF>> particle_adv_density;
  std::unique_ptr<Variable<1, TF3>> particle_pressure_accel;
  std::unique_ptr<Variable<1, TF>> particle_density_err;

  std::unique_ptr<Variable<4, TQ>> pid;
  std::unique_ptr<Variable<3, U>> pid_length;
  std::unique_ptr<Variable<2, TQ>> particle_neighbors;
  std::unique_ptr<Variable<1, U>> particle_num_neighbors;
};
}  // namespace alluvion

#endif /* ALLUVION_SOLVER_II_HPP */
