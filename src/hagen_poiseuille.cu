#include <glm/gtc/type_ptr.hpp>
#include <iostream>

#include "alluvion/constants.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;
using namespace alluvion::dg;

int main(void) {
  Store store;
  Display* display = store.create_display(800, 600, "particle view");

  F particle_radius = 0.025_F;
  F kernel_radius = particle_radius * 4._F;
  F density0 = 1000.0_F;
  F cubical_particle_volume =
      8 * particle_radius * particle_radius * particle_radius;
  F volume_relative_to_cube = 0.8_F;
  F particle_mass =
      cubical_particle_volume * volume_relative_to_cube * density0;

  F cfl_factor = 0.04_F;
  F cfl_length_scale = particle_radius * 2._F;
  F3 gravity = {0._F, -9.81_F, 0._F};
  F max_dt = cfl_factor * cfl_length_scale / length(gravity);
  F dt = max_dt * 0.1_F;

  cnst::set_cubic_discretization_constants();
  cnst::set_kernel_radius(kernel_radius);
  cnst::set_particle_attr(particle_radius, particle_mass, density0);
  cnst::set_gravity(gravity);
  cnst::set_boundary_epsilon(1e-7_F);
  F viscosity = 0.001_F;
  F vorticity = 0.01_F;
  F inertia_inverse = 0.1_F;
  F viscosity_omega = 0.5_F;
  F surface_tension_coeff = 0.05_F;
  F surface_tension_boundary_coeff = 0.01_F;
  cnst::set_advanced_fluid_attr(viscosity, vorticity, inertia_inverse,
                                viscosity_omega, surface_tension_coeff,
                                surface_tension_boundary_coeff);

  I kM = 4;
  F cylinder_length = 2._F * kM * kernel_radius;
  I kQ = 5;
  F R = kernel_radius * kQ;

  display->camera_.setEye(0._F, 0._F, R * 6._F);
  display->camera_.setClipPlanes(particle_radius * 10._F, R * 20._F);
  display->update_trackball_camera();

  // rigids
  F restitution = 1._F;
  F friction = 0._F;
  F boundary_viscosity = viscosity;
  U max_num_contacts = 512;
  Pile pile(store, max_num_contacts);
  pile.add(new InfiniteCylinderDistance(R), U3{1, 20, 20}, -1._F, 0, Mesh(),
           0._F, restitution, friction, boundary_viscosity, F3{1, 1, 1},
           F3{0, 0, 0}, Q{0, 0, 0, 1}, Mesh());
  pile.build_grids(4 * kernel_radius);
  pile.reallocate_kinematics_on_device();
  cnst::set_num_boundaries(pile.get_size());
  cnst::set_contact_tolerance(0.05_F);

  // particles
  U num_particles = 0;
  U max_num_particles =
      static_cast<U>(2._F * kPi<F> * kQ * kQ * kM * kernel_radius *
                     kernel_radius * kernel_radius * density0 / particle_mass);
  GraphicalVariable<1, F3> particle_x =
      store.create_graphical<1, F3>({max_num_particles});
  Variable<1, F3> particle_v = store.create<1, F3>({max_num_particles});
  Variable<1, F3> particle_a = store.create<1, F3>({max_num_particles});
  Variable<1, F> particle_density = store.create<1, F>({max_num_particles});
  Variable<1, F> particle_pressure = store.create<1, F>({max_num_particles});
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
  Variable<1, F3> particle_dij_pj = store.create<1, F3>({max_num_particles});
  Variable<1, F> particle_sum_tmp = store.create<1, F>({max_num_particles});
  Variable<1, F> particle_adv_density = store.create<1, F>({max_num_particles});
  Variable<1, F3> particle_pressure_accel =
      store.create<1, F3>({max_num_particles});
  Variable<1, F> particle_cfl_v2 = store.create<1, F>({max_num_particles});

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
  cnst::set_max_num_neighbors_per_particle(max_num_neighbors_per_particle);
  cnst::set_wrap_length(grid_res.x * kernel_radius);
  Variable<4, U> pid = store.create<4, U>(
      {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell});
  Variable<3, U> pid_length =
      store.create<3, U>({grid_res.x, grid_res.y, grid_res.z});
  // neighbor
  Variable<2, U> particle_neighbors =
      store.create<2, U>({max_num_particles, max_num_neighbors_per_particle});
  Variable<1, U> particle_num_neighbors =
      store.create<1, U>({max_num_particles});

  U initial_num_particles = max_num_particles * 4 / 5;
  F slice_distance = particle_radius * 2._F;
  U num_slices = static_cast<U>(cylinder_length / slice_distance);
  U num_particles_per_slice = initial_num_particles / num_slices;
  initial_num_particles = num_particles_per_slice * num_slices;

  store.map_graphical_pointers();
  Runner::launch(initial_num_particles, 256, [&](U grid_size, U block_size) {
    create_fluid_cylinder<F3, F><<<grid_size, block_size>>>(
        particle_x, initial_num_particles, R - particle_radius * 2._F,
        num_particles_per_slice, particle_radius * 2._F,
        cylinder_length * -0.5_F);
  });
  store.unmap_graphical_pointers();
  num_particles = initial_num_particles;

  // sample points
  U num_samples = 21;
  Variable<1, F3> sample_x = store.create<1, F3>({num_samples});
  Variable<1, F3> sample_data3 = store.create<1, F3>({num_samples});
  Variable<1, F> sample_data = store.create<1, F>({num_samples});
  Variable<2, U> sample_neighbors =
      store.create<2, U>({num_samples, max_num_neighbors_per_particle});
  Variable<1, U> sample_num_neighbors = store.create<1, U>({num_samples});
  Variable<2, F3> sample_boundary_xj =
      store.create<2, F3>({pile.get_size(), num_samples});
  Variable<2, F> sample_boundary_volume =
      store.create<2, F>({pile.get_size(), num_samples});
  {
    std::vector<F3> sample_x_host(num_samples);
    for (I i = 0; i < num_samples; ++i) {
      sample_x_host[i] = F3{
          0._F,
          R * 2._F / (num_samples + 1) * (i - static_cast<I>(num_samples) / 2),
          0._F};
    }
    sample_x.set_bytes(sample_x_host.data());
  }
  std::vector<F3> sample_data3_host(num_samples);
  std::vector<F> sample_data_host(num_samples);

  U frame_id = 0;
  F t = 0;
  U num_emission = static_cast<U>(cylinder_length / (particle_radius * 2._F));
  F next_emission_t = 0;
  // density sampling for the place with emission
  Variable<1, F3> emission_x = store.create<1, F3>({num_emission + 1});
  Variable<1, U> num_emitted = store.create<1, U>({1});
  Variable<1, F> emission_sample_density =
      store.create<1, F>({num_emission + 1});
  Variable<2, U> emission_neighbors =
      store.create<2, U>({num_emission + 1, max_num_neighbors_per_particle});
  Variable<1, U> emission_num_neighbors =
      store.create<1, U>({num_emission + 1});
  {
    std::vector<F3> emission_x_host(num_emission + 1);
    for (I i = 0; i < num_emission; ++i) {
      emission_x_host[i] =
          F3{cylinder_length * -0.5_F + i * particle_radius * 2._F,
             R - particle_radius * 2._F, 0._F};
    }
    emission_x_host[num_emission] = F3{0._F, -R + particle_radius * 2._F, 0._F};
    emission_x.set_bytes(emission_x_host.data());
  }
  std::vector<F> emission_sample_density_host(num_emission + 1);
  bool finished_filling = false;
  bool should_close = false;
  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, [&](ShadingProgram& program, Display& display) {
        // std::cout << "============= frame_id = " << frame_id << std::endl;
        if (should_close) {
          return;
        }

        store.map_graphical_pointers();
        for (U frame_interstep = 0; frame_interstep < 20; ++frame_interstep) {
          if (true) {
            // sampling
            // pile.for_each_rigid([&](U boundary_id,
            //                         Variable<1, F> const& distance_grid,
            //                         Variable<1, F> const& volume_grid,
            //                         F3 const& rigid_x, Q const& rigid_q,
            //                         F3 const& domain_min, F3 const&
            //                         domain_max, U3 const& resolution, F3
            //                         const& cell_size, U num_nodes, F sign, F
            //                         thickness) {
            //   Runner::launch(num_samples, 256, [&](U grid_size, U block_size)
            //   {
            //     compute_sample_boundary<<<grid_size, block_size>>>(
            //         volume_grid, distance_grid, rigid_x, rigid_q,
            //         boundary_id, domain_min, domain_max, resolution,
            //         cell_size, num_nodes, 0, sign, thickness, dt, sample_x,
            //         sample_boundary_xj, sample_boundary_volume, num_samples);
            //   });
            // });
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
            Runner::launch(
                num_emission + 1, 256, [&](U grid_size, U block_size) {
                  make_neighbor_list_wrapped<<<grid_size, block_size>>>(
                      emission_x, particle_x, pid, pid_length,
                      emission_neighbors, emission_num_neighbors,
                      num_emission + 1);
                });
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              compute_density_fluid_wrapped<<<grid_size, block_size>>>(
                  particle_x, particle_neighbors, particle_num_neighbors,
                  particle_density, num_particles);
            });
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              compute_density_boundary<<<grid_size, block_size>>>(
                  particle_x, particle_density, particle_boundary_xj,
                  particle_boundary_volume, num_particles);
            });

            Runner::launch(num_samples, 256, [&](U grid_size, U block_size) {
              sample_fluid_wrapped<<<grid_size, block_size>>>(
                  sample_x, particle_x, particle_density, particle_density,
                  sample_neighbors, sample_num_neighbors, sample_data,
                  num_samples);
            });
            Runner::launch(num_emission + 1, 256,
                           [&](U grid_size, U block_size) {
                             sample_fluid_wrapped<<<grid_size, block_size>>>(
                                 emission_x, particle_x, particle_density,
                                 particle_density, emission_neighbors,
                                 emission_num_neighbors,
                                 emission_sample_density, num_emission + 1);
                           });
            // emission_sample_density.get_bytes(emission_sample_density_host.data());
            // std::cout<<"Emission density: ";
            // for (const F& point_density: emission_sample_density_host){
            //   std::cout<<point_density<<" ";
            // }
            // std::cout<<std::endl;
            // Runner::launch(num_samples, 256, [&](U grid_size, U block_size) {
            //   sample_density_boundary<<<grid_size, block_size>>>(
            //       sample_x, sample_data, sample_boundary_xj,
            //       sample_boundary_volume, num_samples);
            // });
            sample_data.get_bytes(sample_data_host.data());
            // std::cout << "vertical density: ";
            // for (F const& sample_item : sample_data_host) {
            //   std::cout << sample_item << " ";
            // }
            // std::cout << std::endl;
          }
          // std::cout << "num_particles = " << num_particles << "/"
          //           << max_num_particles << std::endl;
          // emission
          F3 emission_velocity = gravity * max_dt * 500.0_F;
          F emission_speed = length(emission_velocity);
          F emission_interval = particle_radius * 2._F / emission_speed;
          if (!finished_filling && t >= next_emission_t &&
              num_particles + num_emission <= max_num_particles) {
            num_emitted.set_zero();
            Runner::launch(num_emission, 256, [&](U grid_size, U block_size) {
              emit_if_density_lower_than_last<<<grid_size, block_size>>>(
                  particle_x, particle_v, emission_x, emission_sample_density,
                  num_emitted, num_emission, num_particles, 0.99_F,
                  emission_velocity);
            });
            U num_emitted_host;
            num_emitted.get_bytes(&num_emitted_host);
            num_particles += num_emitted_host;
            std::cout << "filled " << num_emitted_host << std::endl;
            if (num_emitted_host > 0) {
              next_emission_t = t + emission_interval;
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
            compute_density_fluid_wrapped<<<grid_size, block_size>>>(
                particle_x, particle_neighbors, particle_num_neighbors,
                particle_density, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            compute_density_boundary<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_boundary_xj,
                particle_boundary_volume, num_particles);
          });

          F max_density = Runner::max(particle_density, num_particles);
          F min_density = Runner::min(particle_density, num_particles);
          // std::cout << "particle density: " << min_density << ", " <<
          // max_density
          //           << " t: "<< t<<std::endl;
          if ((max_density - min_density) / density0 < 0.05) {
            F density_diff =
                abs(sample_data_host[0] - sample_data_host[num_samples - 1]) /
                density0;
            if (density_diff < 0.005_F && !finished_filling) {
              finished_filling = true;
              particle_v.set_zero();
              gravity = F3{1.28e-5_F, 0._F, 0._F};
              cnst::set_gravity(gravity);
              std::cout << "finished filling" << std::endl;
            }
          }

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
          // update_dt
          F max_particle_speed =
              sqrt(Runner::max(particle_cfl_v2, num_particles));
          dt = cfl_factor * (cfl_length_scale / max_particle_speed);
          dt = min(max_dt, dt);
          // std::cout << "dt = " << dt << " max speed = " << max_particle_speed
          //           << std::endl;

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
        }
        store.unmap_graphical_pointers();
        frame_id += 1;
      }));

  // {{{
#include "alluvion/glsl/particle.frag"
#include "alluvion/glsl/particle.vert"
  display->add_shading_program(new ShadingProgram(
      kParticleVertexShaderStr, kParticleFragmentShaderStr,
      {"particle_radius", "screen_dimension", "M", "V", "P",
       "camera_worldspace", "material.diffuse", "material.specular",
       "material.shininess", "directional_light.direction",
       "directional_light.ambient", "directional_light.diffuse",
       "directional_light.specular", "point_lights[0].position",
       "point_lights[0].constant", "point_lights[0].linear",
       "point_lights[0].quadratic", "point_lights[0].ambient",
       "point_lights[0].diffuse", "point_lights[0].specular",
       //
       "point_lights[1].position", "point_lights[1].constant",
       "point_lights[1].linear", "point_lights[1].quadratic",
       "point_lights[1].ambient", "point_lights[1].diffuse",
       "point_lights[1].specular"

      },
      [&particle_x, &num_particles, particle_radius, &grid_res, kernel_radius](
          ShadingProgram& program, Display& display) {
        glUniformMatrix4fv(
            program.get_uniform_location("P"), 1, GL_FALSE,
            glm::value_ptr(display.camera_.getProjectionMatrix()));
        glUniformMatrix4fv(program.get_uniform_location("V"), 1, GL_FALSE,
                           glm::value_ptr(display.camera_.getViewMatrix()));
        glUniform2f(program.get_uniform_location("screen_dimension"),
                    static_cast<GLfloat>(display.width_),
                    static_cast<GLfloat>(display.height_));
        glUniform1f(program.get_uniform_location("particle_radius"),
                    particle_radius);

        glm::vec3 const& camera_worldspace = display.camera_.getCenter();
        glUniform3f(program.get_uniform_location("camera_worldspace"),
                    camera_worldspace[0], camera_worldspace[1],
                    camera_worldspace[2]);
        glUniform3f(program.get_uniform_location("directional_light.direction"),
                    0.2f, 1.0f, 0.3f);
        glUniform3f(program.get_uniform_location("directional_light.ambient"),
                    0.05f, 0.05f, 0.05f);
        glUniform3f(program.get_uniform_location("directional_light.diffuse"),
                    0.4f, 0.4f, 0.4f);
        glUniform3f(program.get_uniform_location("directional_light.specular"),
                    0.5f, 0.5f, 0.5f);

        glUniform3f(program.get_uniform_location("point_lights[0].position"),
                    2.0f, 2.0f, 2.0f);
        glUniform1f(program.get_uniform_location("point_lights[0].constant"),
                    1.0f);
        glUniform1f(program.get_uniform_location("point_lights[0].linear"),
                    0.09f);
        glUniform1f(program.get_uniform_location("point_lights[0].quadratic"),
                    0.032f);
        glUniform3f(program.get_uniform_location("point_lights[0].ambient"),
                    0.05f, 0.05f, 0.05f);
        glUniform3f(program.get_uniform_location("point_lights[0].diffuse"),
                    0.8f, 0.8f, 0.8f);
        glUniform3f(program.get_uniform_location("point_lights[0].specular"),
                    1.0f, 1.0f, 1.0f);

        glUniform3f(program.get_uniform_location("point_lights[1].position"),
                    2.0f, 1.0f, -2.0f);
        glUniform1f(program.get_uniform_location("point_lights[1].constant"),
                    1.0f);
        glUniform1f(program.get_uniform_location("point_lights[1].linear"),
                    0.09f);
        glUniform1f(program.get_uniform_location("point_lights[1].quadratic"),
                    0.032f);
        glUniform3f(program.get_uniform_location("point_lights[1].ambient"),
                    0.05f, 0.05f, 0.05f);
        glUniform3f(program.get_uniform_location("point_lights[1].diffuse"),
                    0.8f, 0.8f, 0.8f);
        glUniform3f(program.get_uniform_location("point_lights[1].specular"),
                    1.0f, 1.0f, 1.0f);
        glUniform3f(program.get_uniform_location("material.diffuse"), 0.2f,
                    0.3f, 0.87f);
        glUniform3f(program.get_uniform_location("material.specular"), 0.8f,
                    0.9f, 0.9f);
        glUniform1f(program.get_uniform_location("material.shininess"), 5.0f);

        glBindBuffer(GL_ARRAY_BUFFER, particle_x.vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        for (I i = 0; i <= 0; ++i) {
          float wrap_length = grid_res.x * kernel_radius;
          glUniformMatrix4fv(
              program.get_uniform_location("M"), 1, GL_FALSE,
              glm::value_ptr(glm::translate(glm::mat4(1),
                                            glm::vec3{wrap_length * i, 0, 0})));
          glDrawArrays(GL_POINTS, 0, num_particles);
        }
        glDisableVertexAttribArray(0);
      }));

  display->run();
  // }}}
}