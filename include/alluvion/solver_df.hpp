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
  using Base::cfl;
  using Base::dt;
  using Base::enable_surface_tension;
  using Base::enable_vorticity;
  using Base::max_dt;
  using Base::min_dt;
  using Base::num_particles;
  using Base::particle_radius;
  using Base::pile;
  using Base::runner;
  using Base::sample_usher;
  using Base::t;
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
  SolverDf(TRunner& runner_arg, TPile& pile_arg, Store& store,
           U max_num_particles_arg, U3 grid_res, U num_ushers = 0,
           bool enable_surface_tension_arg = false,
           bool enable_vorticity_arg = false, bool graphical = false)
      : Base(runner_arg, pile_arg, store, max_num_particles_arg, grid_res,
             num_ushers, enable_surface_tension_arg, enable_vorticity_arg,
             graphical),
        particle_dfsph_factor(store.create<1, TF>({max_num_particles_arg})),
        particle_kappa(store.create<1, TF>({max_num_particles_arg})),
        particle_kappa_v(store.create<1, TF>({max_num_particles_arg})),
        particle_density_adv(store.create<1, TF>({max_num_particles_arg})) {}

  template <U wrap, U gravitation>
  void step() {
    particle_force->set_zero();
    particle_torque->set_zero();
    pile.copy_kinematics_to_device();
    pile.for_each_rigid(
        [&](U boundary_id, dg::Distance<TF3, TF> const& distance,
            Variable<1, TF> const& distance_grid,
            Variable<1, TF> const& volume_grid, TF3 const& rigid_x,
            TQ const& rigid_q, TF3 const& domain_min, TF3 const& domain_max,
            U3 const& resolution, TF3 const& cell_size, U num_nodes, TF sign,
            TF thickness) {
          runner.launch_compute_particle_boundary(
              distance, volume_grid, distance_grid, rigid_x, rigid_q,
              boundary_id, domain_min, domain_max, resolution, cell_size,
              num_nodes, 0, sign, thickness, dt, *particle_x,
              *particle_boundary, *particle_boundary_kernel, num_particles);
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
              *particle_density, *particle_boundary_kernel, num_particles);
        },
        "compute_density", compute_density<TQ, TF3, TF>);
    if (usher->num_ushers > 0) {
      sample_usher();
    }
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_dfsph_factor<<<grid_size, block_size>>>(
              *particle_x, *particle_neighbors, *particle_num_neighbors,
              *particle_dfsph_factor, *particle_boundary_kernel, num_particles);
        },
        "compute_dfsph_factor", compute_dfsph_factor<TQ, TF3, TF>);
    // ===== [divergence solve
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          warm_start_divergence_solve_0<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_kappa_v, *particle_neighbors,
              *particle_num_neighbors, *particle_boundary,
              *particle_boundary_kernel, *pile.x_device_, *pile.v_device_,
              *pile.omega_device_, dt, num_particles);
        },
        "warm_start_divergence_solve_0",
        warm_start_divergence_solve_0<TQ, TF3, TF>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          warm_start_divergence_solve_1<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_kappa_v, *particle_neighbors,
              *particle_num_neighbors, *particle_boundary,
              *particle_boundary_kernel, *particle_force, *particle_torque,
              *pile.x_device_, *pile.v_device_, *pile.omega_device_, dt,
              num_particles);
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

    TF avg_density_err = std::numeric_limits<TF>::max();
    constexpr TF kMaxError = 1e-3;
    U num_divergence_solve_iteration = 0;
    while (num_divergence_solve_iteration < 1 ||
           avg_density_err > kMaxError / dt) {
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
                *particle_boundary, *particle_boundary_kernel, *pile.x_device_,
                *pile.v_device_, *pile.omega_device_, dt, num_particles);
          },
          "compute_divergence_solve_density_error",
          compute_divergence_solve_density_error<TQ, TF3, TF>);
      avg_density_err =
          TRunner::sum(*particle_density_adv, num_particles) / num_particles;
      ++num_divergence_solve_iteration;
    }
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          divergence_solve_finish<<<grid_size, block_size>>>(
              *particle_dfsph_factor, *particle_kappa_v, dt, num_particles);
        },
        "divergence_solve_finish", divergence_solve_finish<TF>);
    // ===== ]divergence solve

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

    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          integrate_non_pressure_acceleration<<<grid_size, block_size>>>(
              *particle_v, *particle_a, dt, num_particles);
        },
        "integrate_non_pressure_acceleration",
        integrate_non_pressure_acceleration<TF3, TF>);
    // ===== [pressure solve
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          warm_start_pressure_solve0<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_density, *particle_kappa,
              *particle_neighbors, *particle_num_neighbors, *particle_boundary,
              *particle_boundary_kernel, *pile.x_device_, *pile.v_device_,
              *pile.omega_device_, dt, num_particles);
        },
        "warm_start_pressure_solve0", warm_start_pressure_solve0<TQ, TF3, TF>);
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
        "warm_start_pressure_solve1", warm_start_pressure_solve1<TQ, TF3, TF>);
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_rho_adv<<<grid_size, block_size>>>(
              *particle_x, *particle_v, *particle_density,
              *particle_dfsph_factor, *particle_density_adv,
              *particle_neighbors, *particle_num_neighbors, *particle_boundary,
              *particle_boundary_kernel, *pile.x_device_, *pile.v_device_,
              *pile.omega_device_, dt, num_particles);
        },
        "compute_rho_adv", compute_rho_adv<TQ, TF3, TF>);
    avg_density_err = std::numeric_limits<TF>::max();
    U num_pressure_solve_iteration = 0;
    while (num_pressure_solve_iteration < 2 || avg_density_err > kMaxError) {
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
      avg_density_err =
          TRunner::sum(*particle_density_adv, num_particles) / num_particles -
          1;  // TODO: add variable to store *particle_density_adv - 1
              // (better numerical precision)
      ++num_pressure_solve_iteration;
    }
    runner.launch(
        num_particles,
        [&](U grid_size, U block_size) {
          pressure_solve_finish<<<grid_size, block_size>>>(*particle_kappa, dt,
                                                           num_particles);
        },
        "pressure_solve_finish", pressure_solve_finish<TF>);
    // ===== ]pressure solve
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
};
}  // namespace alluvion

#endif /* ALLUVION_SOLVER_DF_HPP */
