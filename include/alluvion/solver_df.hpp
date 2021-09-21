#ifndef ALLUVION_SOLVER_DF_HPP
#define ALLUVION_SOLVER_DF_HPP

#include <memory>

#include "alluvion/solver.hpp"
#include "alluvion/store.hpp"
namespace alluvion {
template <typename TF>
struct SolverDf : public Solver<TF> {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
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
  using Base::sample_usher;
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

  // TODO: get grid_res from store.get_cni()
  SolverDf(TRunner& runner_arg, TPile& pile_arg, Store& store_arg,
           U max_num_particles_arg, U3 grid_res, U num_ushers = 0,
           bool enable_surface_tension_arg = false,
           bool enable_vorticity_arg = false, bool graphical = false)
      : Base(runner_arg, pile_arg, store_arg, max_num_particles_arg, grid_res,
             num_ushers, enable_surface_tension_arg, enable_vorticity_arg,
             graphical),
        particle_dfsph_factor(store_arg.create<1, TF>({max_num_particles_arg})),
        particle_kappa(store_arg.create<1, TF>({max_num_particles_arg})),
        particle_kappa_v(store_arg.create<1, TF>({max_num_particles_arg})),
        particle_density_adv(store_arg.create<1, TF>({max_num_particles_arg})),
        enable_divergence_solve(true),
        enable_density_solve(true),
        density_change_tolerance(1e-3),
        density_error_tolerance(1e-3),
        min_density_solve(2),
        max_density_solve(100),
        min_divergence_solve(1),
        max_divergence_solve(100) {}
  virtual ~SolverDf() {
    store.remove(*particle_dfsph_factor);
    store.remove(*particle_kappa);
    store.remove(*particle_kappa_v);
    store.remove(*particle_density_adv);
  }
  template <U wrap, U gravitation>
  void step() {
    particle_force->set_zero();
    particle_torque->set_zero();
    pile.copy_kinematics_to_device();
    compute_all_boundaries();
    Base::template update_particle_neighbors<wrap>();
    if (usher->num_ushers > 0) {
      sample_usher();
    }
    if (enable_divergence_solve) {
      // ===== [divergence solve
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_dfsph_factor<<<grid_size, block_size>>>(
                *particle_x, *particle_neighbors, *particle_num_neighbors,
                *particle_dfsph_factor, *particle_boundary_kernel,
                num_particles);
          },
          "compute_dfsph_factor", compute_dfsph_factor<TQ, TF3, TF>);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            warm_start_divergence_solve_0<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_kappa_v,
                *particle_neighbors, *particle_num_neighbors,
                *particle_boundary, *particle_boundary_kernel, *pile.x_device_,
                *pile.v_device_, *pile.omega_device_, dt, num_particles);
          },
          "warm_start_divergence_solve_0",
          warm_start_divergence_solve_0<TQ, TF3, TF>);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            warm_start_divergence_solve_1<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_kappa_v,
                *particle_neighbors, *particle_num_neighbors,
                *particle_boundary, *particle_boundary_kernel, *particle_force,
                *particle_torque, *pile.x_device_, *pile.v_device_,
                *pile.omega_device_, dt, num_particles);
          },
          "warm_start_divergence_solve_1",
          warm_start_divergence_solve_1<TQ, TF3, TF>);

      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_velocity_of_density_change<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_dfsph_factor,
                *particle_density_adv, *particle_neighbors,
                *particle_num_neighbors, *particle_boundary,
                *particle_boundary_kernel, *pile.x_device_, *pile.v_device_,
                *pile.omega_device_, dt, num_particles);
          },
          "compute_velocity_of_density_change",
          compute_velocity_of_density_change<TQ, TF3, TF>);

      mean_density_change = std::numeric_limits<TF>::max();
      num_divergence_solve = 0;
      while ((num_divergence_solve < min_divergence_solve ||
              mean_density_change > density_change_tolerance / dt) &&
             num_divergence_solve <= max_divergence_solve) {
        runner.launch(
            num_particles,
            [&](U grid_size, U block_size) {
              divergence_solve_iteration<<<grid_size, block_size>>>(
                  *particle_x, *particle_v, *particle_dfsph_factor,
                  *particle_density_adv, *particle_kappa_v, *particle_neighbors,
                  *particle_num_neighbors, *particle_boundary,
                  *particle_boundary_kernel, *particle_force, *particle_torque,
                  *pile.x_device_, *pile.v_device_, *pile.omega_device_, dt,
                  num_particles);
            },
            "divergence_solve_iteration",
            divergence_solve_iteration<TQ, TF3, TF>);

        runner.launch(
            num_particles,
            [&](U grid_size, U block_size) {
              compute_divergence_solve_density_error<<<grid_size, block_size>>>(
                  *particle_x, *particle_v, *particle_density_adv,
                  *particle_neighbors, *particle_num_neighbors,
                  *particle_boundary, *particle_boundary_kernel,
                  *pile.x_device_, *pile.v_device_, *pile.omega_device_, dt,
                  num_particles);
            },
            "compute_divergence_solve_density_error",
            compute_divergence_solve_density_error<TQ, TF3, TF>);
        mean_density_change =
            TRunner::sum(*particle_density_adv, num_particles) / num_particles;
        ++num_divergence_solve;
      }
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            divergence_solve_finish<<<grid_size, block_size>>>(
                *particle_dfsph_factor, *particle_kappa_v, dt, num_particles);
          },
          "divergence_solve_finish", divergence_solve_finish<TF>);
      // ===== ]divergence solve
    }

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
            compute_vorticity<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_density, *particle_omega,
                *particle_a, *particle_angular_acceleration,
                *particle_neighbors, *particle_num_neighbors, *particle_force,
                *particle_torque, *particle_boundary, *particle_boundary_kernel,
                *pile.x_device_, *pile.v_device_, *pile.omega_device_, dt,
                num_particles);
          },
          "compute_vorticity", compute_vorticity<TQ, TF3, TF>);
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
            drive_linear<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_a, *(usher->drive_x),
                *(usher->drive_v), *(usher->drive_kernel_radius),
                *(usher->drive_strength), usher->num_ushers, num_particles);
          },
          "drive_linear", drive_linear<TF3, TF>);
    }
    update_dt();

    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          integrate_non_pressure_acceleration<<<grid_size, block_size>>>(
              *particle_v, *particle_a, dt, num_particles);
        },
        "integrate_non_pressure_acceleration",
        integrate_non_pressure_acceleration<TF3, TF>);
    if (enable_density_solve) {
      // ===== [density solve
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            warm_start_pressure_solve0<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_density, *particle_kappa,
                *particle_neighbors, *particle_num_neighbors,
                *particle_boundary, *particle_boundary_kernel, *pile.x_device_,
                *pile.v_device_, *pile.omega_device_, dt, num_particles);
          },
          "warm_start_pressure_solve0",
          warm_start_pressure_solve0<TQ, TF3, TF>);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            warm_start_pressure_solve1<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_kappa, *particle_neighbors,
                *particle_num_neighbors, *particle_boundary,
                *particle_boundary_kernel, *particle_force, *particle_torque,
                *pile.x_device_, *pile.v_device_, *pile.omega_device_, dt,
                num_particles);
          },
          "warm_start_pressure_solve1",
          warm_start_pressure_solve1<TQ, TF3, TF>);
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_rho_adv<<<grid_size, block_size>>>(
                *particle_x, *particle_v, *particle_density,
                *particle_dfsph_factor, *particle_density_adv,
                *particle_neighbors, *particle_num_neighbors,
                *particle_boundary, *particle_boundary_kernel, *pile.x_device_,
                *pile.v_device_, *pile.omega_device_, dt, num_particles);
          },
          "compute_rho_adv", compute_rho_adv<TQ, TF3, TF>);
      mean_density_error = std::numeric_limits<TF>::max();
      num_density_solve = 0;
      while ((num_density_solve < min_density_solve ||
              mean_density_error > density_error_tolerance) &&
             num_density_solve <= max_density_solve) {
        runner.launch(
            num_particles,
            [&](U grid_size, U block_size) {
              pressure_solve_iteration<<<grid_size, block_size>>>(
                  *particle_x, *particle_v, *particle_dfsph_factor,
                  *particle_density_adv, *particle_kappa, *particle_neighbors,
                  *particle_num_neighbors, *particle_boundary,
                  *particle_boundary_kernel, *particle_force, *particle_torque,
                  *pile.x_device_, *pile.v_device_, *pile.omega_device_, dt,
                  num_particles);
            },
            "pressure_solve_iteration", pressure_solve_iteration<TQ, TF3, TF>);
        runner.launch(
            num_particles,
            [&](U grid_size, U block_size) {
              compute_pressure_solve_density_error<<<grid_size, block_size>>>(
                  *particle_x, *particle_v, *particle_density,
                  *particle_density_adv, *particle_neighbors,
                  *particle_num_neighbors, *particle_boundary,
                  *particle_boundary_kernel, *pile.x_device_, *pile.v_device_,
                  *pile.omega_device_, dt, num_particles);
            },
            "compute_pressure_solve_density_error",
            compute_pressure_solve_density_error<TQ, TF3, TF>);
        mean_density_error =
            TRunner::sum(*particle_density_adv, num_particles) / num_particles -
            1;  // TODO: add variable to store *particle_density_adv - 1
                // (better numerical precision)
        ++num_density_solve;
      }
      runner.launch(
          num_particles,
          [&](U grid_size, U block_size) {
            pressure_solve_finish<<<grid_size, block_size>>>(*particle_kappa,
                                                             dt, num_particles);
          },
          "pressure_solve_finish", pressure_solve_finish<TF>);
      // ===== ]density solve
    }
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          integrate_velocity<wrap><<<grid_size, block_size>>>(
              *particle_x, *particle_v, dt, num_particles);
        },
        "integrate_velocity", integrate_velocity<wrap, TF3, TF>);

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
    particle_kappa->set_zero();
    particle_kappa_v->set_zero();
  }

  std::unique_ptr<Variable<1, TF>> particle_dfsph_factor;
  std::unique_ptr<Variable<1, TF>> particle_kappa;
  std::unique_ptr<Variable<1, TF>> particle_kappa_v;
  std::unique_ptr<Variable<1, TF>> particle_density_adv;

  bool enable_divergence_solve;
  bool enable_density_solve;
  TF density_change_tolerance;
  TF density_error_tolerance;
  U min_density_solve;
  U max_density_solve;
  U min_divergence_solve;
  U max_divergence_solve;

  // report
  U num_divergence_solve;
  U num_density_solve;
  TF mean_density_change;
  TF mean_density_error;
};
}  // namespace alluvion

#endif /* ALLUVION_SOLVER_DF_HPP */
