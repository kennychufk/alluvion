#ifndef ALLUVION_SOLVER_I_HPP
#define ALLUVION_SOLVER_I_HPP

#include "alluvion/solver.hpp"
#include "alluvion/store.hpp"
namespace alluvion {
template <typename TF>
struct SolverI : public Solver<TF> {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float2, double2> TF2;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  using Base = Solver<TF>;
  using Base::compute_all_boundaries;
  using Base::dt;
  using Base::enable_surface_tension;
  using Base::enable_vorticity;
  using Base::num_particles;
  using Base::particle_radius;
  using Base::pile;
  using Base::runner;
  using Base::store;
  using Base::t;
  using Base::update_dt;
  using Base::usher;

  using Base::particle_a;
  using Base::particle_angular_acceleration;
  using Base::particle_boundary;
  using Base::particle_boundary_kernel;
  using Base::particle_cfl_v2;
  using Base::particle_density;
  using Base::particle_force;
  using Base::particle_normal;
  using Base::particle_omega;
  using Base::particle_torque;
  using Base::particle_v;
  using Base::particle_x;

  using Base::particle_neighbors;
  using Base::particle_num_neighbors;
  using Base::pid;
  using Base::pid_length;
  SolverI(TRunner& runner_arg, TPile& pile_arg, Store& store_arg,
          U max_num_particles_arg, U3 grid_res, U num_ushers,
          bool enable_surface_tension_arg = false,
          bool enable_vorticity_arg = false, bool graphical = false)
      : Base(runner_arg, pile_arg, store_arg, max_num_particles_arg, grid_res,
             num_ushers, enable_surface_tension_arg, enable_vorticity_arg,
             graphical),
        particle_pressure(store_arg.create<1, TF>({max_num_particles_arg})),
        particle_last_pressure(
            store_arg.create<1, TF>({max_num_particles_arg})),
        particle_diag_adv_density(
            store_arg.create<1, TF2>({max_num_particles_arg})),
        particle_pressure_accel(
            store_arg.create<1, TF3>({max_num_particles_arg})),
        particle_density_err(store_arg.create<1, TF>({max_num_particles_arg})),
        density_error_tolerance(1e-3),
        min_density_solve(2),
        max_density_solve(100) {}
  virtual ~SolverI() {
    store.remove(*particle_pressure);
    store.remove(*particle_last_pressure);
    store.remove(*particle_diag_adv_density);
    store.remove(*particle_pressure_accel);
    store.remove(*particle_density_err);
  }
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
    compute_all_boundaries();
    Base::template update_particle_neighbors<wrap>();
    if (enable_surface_tension) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_normal<<<grid_size, block_size>>>(
                *particle_x, *particle_density, *particle_normal,
                *particle_neighbors, *particle_num_neighbors, num_particles);
          },
          "compute_normal", compute_normal<TQ, TF3, TF>);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_surface_tension<<<grid_size, block_size>>>(
                *particle_x, *particle_density, *particle_normal, *particle_a,
                *particle_neighbors, *particle_num_neighbors,
                *particle_boundary, num_particles);
          },
          "compute_surface_tension", compute_surface_tension<TQ, TF3, TF>);
    }
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_viscosity<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_density, *particle_neighbors,
              *particle_num_neighbors, *particle_a, *particle_force,
              *particle_torque, *particle_boundary, *pile.x_device_,
              *pile.v_device_, *pile.omega_device_, num_particles);
        },
        "compute_viscosity", compute_viscosity<TQ, TF3, TF>);
    if (enable_vorticity) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_micropolar_vorticity<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_density, *particle_omega,
                *particle_a, *particle_angular_acceleration,
                *particle_neighbors, *particle_num_neighbors, *particle_force,
                *particle_torque, *particle_boundary, *particle_boundary_kernel,
                *pile.x_device_, *pile.v_device_, *pile.omega_device_, dt,
                num_particles);
          },
          "compute_micropolar_vorticity",
          compute_micropolar_vorticity<TQ, TF3, TF>);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            integrate_angular_acceleration<<<grid_size, block_size>>>(
                *particle_omega, *particle_angular_acceleration, dt,
                num_particles);
          },
          "integrate_angular_acceleration",
          integrate_angular_acceleration<TF3, TF>);
    }
    if (usher->num_ushers > 0) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            drive_n_ellipse<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_a, *(usher->focal_x),
                *(usher->focal_v), *(usher->focal_dist),
                *(usher->usher_kernel_radius), *(usher->drive_strength),
                usher->num_ushers, num_particles);
          },
          "drive_n_ellipse", drive_n_ellipse<TF3, TF>);
    }
    update_dt();

    // ===== [solve
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          advect_and_init_pressure<<<grid_size, block_size>>>(
              *particle_v, *particle_a, *particle_pressure,
              *particle_last_pressure, dt, num_particles);
        },
        "advect_and_init_pressure", advect_and_init_pressure<TF3, TF>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          calculate_isph_diagonal_adv_density<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_density,
              *particle_diag_adv_density, *particle_neighbors,
              *particle_num_neighbors, *particle_boundary,
              *particle_boundary_kernel, *pile.x_device_, *pile.v_device_,
              *pile.omega_device_, dt, num_particles);
        },
        "calculate_isph_diagonal_adv_density",
        calculate_isph_diagonal_adv_density<TQ, TF3, TF2, TF>);
    mean_density_error = std::numeric_limits<TF>::max();
    num_density_solve = 0;
    while ((mean_density_error > density_error_tolerance ||
            num_density_solve < min_density_solve) &&
           num_density_solve < max_density_solve) {
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            isph_solve_iteration<<<grid_size, block_size>>>(
                *particle_x, *particle_density, *particle_last_pressure,
                *particle_pressure, *particle_diag_adv_density,
                *particle_density_err, *particle_neighbors,
                *particle_num_neighbors, dt, num_particles);
          },
          "isph_solve_iteration", isph_solve_iteration<TQ, TF3, TF2, TF>);
      particle_last_pressure->set_from(*particle_pressure);
      mean_density_error =
          TRunner::sum(*particle_density_err, num_particles) / num_particles;
      ++num_density_solve;
    }
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_pressure_accels<<<grid_size, block_size>>>(
              *particle_x, *particle_density, *particle_pressure,
              *particle_pressure_accel, *particle_neighbors,
              *particle_num_neighbors, *particle_force, *particle_torque,
              *particle_boundary, *particle_boundary_kernel, *pile.x_device_,
              num_particles);
        },
        "compute_pressure_accels", compute_pressure_accels<TQ, TF3, TF>);
    // ===== ]solve
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

  virtual void reset_solving_var() override {
    Base::reset_solving_var();
    particle_pressure->set_zero();
  }

  std::unique_ptr<Variable<1, TF>> particle_pressure;
  std::unique_ptr<Variable<1, TF>> particle_last_pressure;
  std::unique_ptr<Variable<1, TF2>> particle_diag_adv_density;
  std::unique_ptr<Variable<1, TF3>> particle_pressure_accel;
  std::unique_ptr<Variable<1, TF>> particle_density_err;

  TF density_error_tolerance;
  U min_density_solve;
  U max_density_solve;

  // report
  U num_density_solve;
  TF mean_density_error;
};
}  // namespace alluvion

#endif /* ALLUVION_SOLVER_I_HPP */