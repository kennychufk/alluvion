#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "alluvion/constants.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;
using namespace alluvion::dg;
namespace py = pybind11;

template <unsigned int D, typename M>
void declare_variable(py::module& m, py::class_<Store>& store_class,
                      const char* name) {
  using Class = Variable<D, M>;
  std::string create_func_name = std::string("create") + name;
  store_class.def(create_func_name.c_str(), &Store::create<D, M>);

  std::string variable_name = std::string("Variable") + name;
  py::class_<Class>(m, variable_name.c_str())
      .def(py::init<const Class&>())
      .def("get_bytes",
           [](Class& variable, py::array_t<unsigned char> bytes) {
             variable.get_bytes(bytes.mutable_data(), bytes.size());
           })
      .def("set_bytes",
           [](Class& variable, py::array_t<unsigned char> bytes) {
             variable.set_bytes(bytes.data(), bytes.size());
           })
      .def("get_type", &Class::get_type)
      .def("get_num_primitives_per_unit", &Class::get_num_primitives_per_unit)
      .def("get_linear_shape", &Class::get_linear_shape)
      .def("get_num_primitives", &Class::get_num_primitives)
      .def("get_shape", &Class::get_shape);
}

PYBIND11_MODULE(_alluvion, m) {
  m.doc() = "CUDA backend for SPH fluid simulation.";
  using namespace pybind11;

  py::class_<Store> store_class =
      py::class_<Store>(m, "Store").def(py::init<>());

  declare_variable<1, F>(m, store_class, "1DF");
  declare_variable<1, F3>(m, store_class, "1DF3");
  declare_variable<2, F3>(m, store_class, "2DF3");

  // py::class_<Runner>(m, "Runner")
  //     .def_static(
  //         "sum", [](py::array_t<unsigned char> dst, Variable<D, F3>& var)
  //         {
  //           Runner::sum<F>(reinterpret_cast<void*>(dst.mutable_data()),
  //                          var.ptr_, var.get_num_primitives(), var.type_);
  //         });
  py::enum_<NumericType>(m, "NumericType")
      .value("f32", NumericType::f32)
      .value("f64", NumericType::f64)
      .value("i32", NumericType::i32)
      .value("u32", NumericType::u32)
      .value("undefined", NumericType::undefined);
  py::class_<Display>(m, "Display").def(py::init<int, int, const char*>());

  m.def(
      "evaluate_developing_hagen_poiseuille",
      [](I kQ, std::string particle_x_filename, F pressure_gradient_acc_x,
         F viscosity, F boundary_viscosity, F dt,
         std::vector<F> const& sample_ts) -> std::vector<F> {
        Store store;

        F particle_radius = 0.0025_F;
        F kernel_radius = particle_radius * 4.0_F;
        F density0 = 1000.0_F;
        F cubical_particle_volume =
            8 * particle_radius * particle_radius * particle_radius;
        F volume_relative_to_cube = 0.8_F;
        F particle_mass =
            cubical_particle_volume * volume_relative_to_cube * density0;

        F3 pressure_gradient_acc = F3{pressure_gradient_acc_x, 0._F, 0._F};

        cnst::set_cubic_discretization_constants();
        cnst::set_kernel_radius(kernel_radius);
        cnst::set_particle_attr(particle_radius, particle_mass, density0);
        cnst::set_gravity(pressure_gradient_acc);
        cnst::set_boundary_epsilon(1e-9_F);
        F vorticity = 0.01_F;
        F inertia_inverse = 0.1_F;
        F viscosity_omega = 0.5_F;
        F surface_tension_coeff = 0.05_F;
        F surface_tension_boundary_coeff = 0.01_F;
        cnst::set_advanced_fluid_attr(viscosity, vorticity, inertia_inverse,
                                      viscosity_omega, surface_tension_coeff,
                                      surface_tension_boundary_coeff);

        I kM = 2;
        F cylinder_length = 2._F * kM * kernel_radius;
        F R = kernel_radius * kQ;

        // rigids
        F restitution = 1._F;
        F friction = 0._F;
        U max_num_contacts = 512;
        Pile pile(store, max_num_contacts);
        pile.add(new InfiniteCylinderDistance(R), U3{1, 20, 20}, -1._F, 0,
                 Mesh(), 0._F, restitution, friction, boundary_viscosity,
                 F3{1, 1, 1}, F3{0, 0, 0}, Q{0, 0, 0, 1}, Mesh());
        pile.build_grids(4 * kernel_radius);
        pile.reallocate_kinematics_on_device();
        cnst::set_num_boundaries(pile.get_size());
        cnst::set_contact_tolerance(0.05_F);

        // particles
        U num_particles = 0;
        U max_num_particles = static_cast<U>(
            2._F * kPi<F> * kQ * kQ * kM * kernel_radius * kernel_radius *
            kernel_radius * density0 / particle_mass);
        Variable<1, F3> particle_x = store.create<1, F3>({max_num_particles});
        Variable<1, F3> particle_v = store.create<1, F3>({max_num_particles});
        Variable<1, F3> particle_a = store.create<1, F3>({max_num_particles});
        Variable<1, F> particle_density =
            store.create<1, F>({max_num_particles});
        Variable<1, F> particle_pressure =
            store.create<1, F>({max_num_particles});
        Variable<1, F> particle_last_pressure =
            store.create<1, F>({max_num_particles});
        Variable<2, F3> particle_boundary_xj =
            store.create<2, F3>({pile.get_size(), max_num_particles});
        Variable<2, F> particle_boundary_volume =
            store.create<2, F>({pile.get_size(), max_num_particles});
        Variable<2, F3> particle_force =
            store.create<2, F3>({pile.get_size(), max_num_particles});
        Variable<2, F3> particle_torque =
            store.create<2, F3>({pile.get_size(), max_num_particles});
        Variable<1, F> particle_aii = store.create<1, F>({max_num_particles});
        Variable<1, F3> particle_dii = store.create<1, F3>({max_num_particles});
        Variable<1, F3> particle_dij_pj =
            store.create<1, F3>({max_num_particles});
        Variable<1, F> particle_sum_tmp =
            store.create<1, F>({max_num_particles});
        Variable<1, F> particle_adv_density =
            store.create<1, F>({max_num_particles});
        Variable<1, F3> particle_pressure_accel =
            store.create<1, F3>({max_num_particles});
        Variable<1, F> particle_cfl_v2 =
            store.create<1, F>({max_num_particles});

        // grid
        U3 grid_res{static_cast<U>(kM * 2), static_cast<U>(kQ * 2),
                    static_cast<U>(kQ * 2)};
        I3 grid_offset{-kM, -kQ, -kQ};
        U max_num_particles_per_cell = 128;
        U max_num_neighbors_per_particle = 128;
        cnst::init_grid_constants(grid_res, grid_offset);
        cnst::set_cell_width(kernel_radius);
        cnst::set_search_range(2.5_F);
        cnst::set_max_num_particles_per_cell(max_num_particles_per_cell);
        cnst::set_max_num_neighbors_per_particle(
            max_num_neighbors_per_particle);
        cnst::set_wrap_length(grid_res.x * kernel_radius);
        Variable<4, U> pid = store.create<4, U>(
            {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell});
        Variable<3, U> pid_length =
            store.create<3, U>({grid_res.x, grid_res.y, grid_res.z});
        // neighbor
        Variable<2, U> particle_neighbors = store.create<2, U>(
            {max_num_particles, max_num_neighbors_per_particle});
        Variable<1, U> particle_num_neighbors =
            store.create<1, U>({max_num_particles});

        // sample points
        U num_sample_planes = 14;
        U num_samples_per_plane = 31;
        U num_samples = num_samples_per_plane * num_sample_planes;
        Variable<1, F3> sample_x = store.create<1, F3>({num_samples});
        Variable<1, F3> sample_data3 = store.create<1, F3>({num_samples});
        Variable<2, U> sample_neighbors =
            store.create<2, U>({num_samples, max_num_neighbors_per_particle});
        Variable<1, U> sample_num_neighbors = store.create<1, U>({num_samples});
        Variable<2, F3> sample_boundary_xj =
            store.create<2, F3>({pile.get_size(), num_samples});
        Variable<2, F> sample_boundary_volume =
            store.create<2, F>({pile.get_size(), num_samples});
        {
          std::vector<F3> sample_x_host(num_samples);
          F distance_between_sample_planes =
              cylinder_length / num_sample_planes;
          for (I i = 0; i < num_samples; ++i) {
            I plane_id = i / num_samples_per_plane;
            I id_in_plane = i % num_samples_per_plane;
            sample_x_host[i] = F3{
                cylinder_length * -0.5_F +
                    distance_between_sample_planes * plane_id,
                R * 2._F / (num_samples_per_plane + 1) *
                    (id_in_plane - static_cast<I>(num_samples_per_plane) / 2),
                0._F};
          }
          sample_x.set_bytes(sample_x_host.data());
        }
        std::vector<F3> sample_data3_host(num_samples);
        std::vector<F> vx(num_samples_per_plane * sample_ts.size());

        U step_id = 0;
        F t = 0;

        I sampling_cursor = 0;

        num_particles = particle_x.read_file(particle_x_filename.c_str());

        while (true) {
          if (t >= sample_ts[sampling_cursor]) {
            pid_length.set_zero();
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              update_particle_grid<<<grid_size, block_size>>>(
                  particle_x, pid, pid_length, num_particles);
            });
            Runner::launch(num_samples, 256, [&](U grid_size, U block_size) {
              make_neighbor_list_wrapped<<<grid_size, block_size>>>(
                  sample_x, particle_x, pid, pid_length, sample_neighbors,
                  sample_num_neighbors, num_samples);
            });
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              compute_density_wrapped<<<grid_size, block_size>>>(
                  particle_x, particle_neighbors, particle_num_neighbors,
                  particle_density, particle_boundary_xj,
                  particle_boundary_volume, num_particles);
            });
            Runner::launch(num_samples, 256, [&](U grid_size, U block_size) {
              sample_fluid_wrapped<<<grid_size, block_size>>>(
                  sample_x, particle_x, particle_density, particle_v,
                  sample_neighbors, sample_num_neighbors, sample_data3,
                  num_samples);
            });
            sample_data3.get_bytes(sample_data3_host.data());
            for (I i = 0; i < sample_data3_host.size(); ++i) {
              vx[num_samples_per_plane * sampling_cursor +
                 (i % num_samples_per_plane)] +=
                  sample_data3_host[i].x / num_sample_planes;
            }
            ++sampling_cursor;
            if (sampling_cursor >= sample_ts.size()) {
              return vx;
            }
          }

          particle_force.set_zero();
          particle_torque.set_zero();
          pile.copy_kinematics_to_device();
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            clear_acceleration<<<grid_size, block_size>>>(particle_a,
                                                          num_particles);
          });
          pile.for_each_rigid([&](U boundary_id,
                                  Variable<1, F> const& distance_grid,
                                  Variable<1, F> const& volume_grid,
                                  F3 const& rigid_x, Q const& rigid_q,
                                  F3 const& domain_min, F3 const& domain_max,
                                  U3 const& resolution, F3 const& cell_size,
                                  U num_nodes, F sign, F thickness) {
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              compute_particle_boundary_wrapped<<<grid_size, block_size>>>(
                  volume_grid, distance_grid, rigid_x, rigid_q, boundary_id,
                  domain_min, domain_max, resolution, cell_size, num_nodes, 0,
                  sign, thickness, dt, particle_x, particle_v,
                  particle_boundary_xj, particle_boundary_volume,
                  num_particles);
            });
          });
          pid_length.set_zero();
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            update_particle_grid<<<grid_size, block_size>>>(
                particle_x, pid, pid_length, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            make_neighbor_list_wrapped<<<grid_size, block_size>>>(
                particle_x, particle_x, pid, pid_length, particle_neighbors,
                particle_num_neighbors, num_particles);
          });

          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            compute_density_wrapped<<<grid_size, block_size>>>(
                particle_x, particle_neighbors, particle_num_neighbors,
                particle_density, particle_boundary_xj,
                particle_boundary_volume, num_particles);
          });

          // compute_normal
          // compute_surface_tension_fluid
          // compute_surface_tension_boundary

          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            compute_viscosity_fluid_wrapped<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_density, particle_neighbors,
                particle_num_neighbors, particle_a, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            compute_viscosity_boundary<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_a, particle_force,
                particle_torque, particle_boundary_xj, particle_boundary_volume,
                pile.x_device_, pile.v_device_, pile.omega_device_,
                pile.boundary_viscosity_device_, num_particles);
          });

          // reset_angular_acceleration
          // compute_vorticity_fluid
          // compute_vorticity_boundary
          // integrate_angular_acceleration
          //
          // calculate_cfl_v2
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            calculate_cfl_v2<<<grid_size, block_size>>>(
                particle_v, particle_a, particle_cfl_v2, dt, num_particles);
          });

          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            predict_advection0_fluid_advect<<<grid_size, block_size>>>(
                particle_v, particle_a, dt, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            predict_advection0_fluid_wrapped<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_dii, particle_neighbors,
                particle_num_neighbors, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            predict_advection0_boundary<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_dii,
                particle_boundary_xj, particle_boundary_volume, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            reset_last_pressure<<<grid_size, block_size>>>(
                particle_pressure, particle_last_pressure, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            predict_advection1_fluid_wrapped<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_dii, particle_adv_density,
                particle_aii, particle_density, particle_neighbors,
                particle_num_neighbors, dt, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            predict_advection1_boundary<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_density, particle_dii,
                particle_adv_density, particle_aii, particle_boundary_xj,
                particle_boundary_volume, pile.x_device_, pile.v_device_,
                pile.omega_device_, dt, num_particles);
          });

          for (U p_solve_iteration = 0; p_solve_iteration < 4;
               ++p_solve_iteration) {
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              pressure_solve_iteration0_wrapped<<<grid_size, block_size>>>(
                  particle_x, particle_density, particle_last_pressure,
                  particle_dij_pj, particle_neighbors, particle_num_neighbors,
                  num_particles);
            });
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              pressure_solve_iteration1_fluid_wrapped<<<grid_size,
                                                        block_size>>>(
                  particle_x, particle_density, particle_last_pressure,
                  particle_dii, particle_dij_pj, particle_sum_tmp,
                  particle_neighbors, particle_num_neighbors, num_particles);
            });
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              pressure_solve_iteration1_boundary<<<grid_size, block_size>>>(
                  particle_x, particle_dij_pj, particle_sum_tmp,
                  particle_boundary_xj, particle_boundary_volume,
                  num_particles);
            });
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              pressure_solve_iteration1_summarize<<<grid_size, block_size>>>(
                  particle_aii, particle_adv_density, particle_sum_tmp,
                  particle_last_pressure, particle_pressure, dt, num_particles);
            });
          }

          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            compute_pressure_accels_fluid_wrapped<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_pressure,
                particle_pressure_accel, particle_neighbors,
                particle_num_neighbors, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            compute_pressure_accels_boundary<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_pressure,
                particle_pressure_accel, particle_force, particle_torque,
                particle_boundary_xj, particle_boundary_volume, pile.x_device_,
                num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            kinematic_integration_wrapped<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_pressure_accel, dt,
                num_particles);
          });
          t += dt;
          step_id += 1;
        }

        return vx;
      });
}
