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

  F particle_radius = 0.25;
  F kernel_radius = 1.0;
  F density0 = 1.0;
  // F particle_mass = 4._F / 3._F * kPi<F> * particle_radius * particle_radius
  // *
  //                   particle_radius * density0;
  F cubical_particle_volume =
      8 * particle_radius * particle_radius * particle_radius;
  F volume_relative_to_cube = 1.0_F;
  F particle_mass =
      cubical_particle_volume * volume_relative_to_cube * density0;
  // F dt = 1e-3;
  F dt = 1e-4;
  F3 gravity = {0._F, -9.81_F, 0._F};
  cnst::set_cubic_discretization_constants();
  cnst::set_kernel_radius(kernel_radius);
  cnst::set_particle_attr(particle_radius, particle_mass, density0);
  cnst::set_gravity(gravity);
  cnst::set_advanced_fluid_attr(0.1, 0.01, 0.1, 0.5, 0.05, 0.01);

  I kM = 8;
  F cylinder_length = kM * kernel_radius;
  I kQ = 10;
  F R = particle_radius * 2._F * kQ;

  // rigids
  U max_num_contacts = 512;
  Pile pile(store, max_num_contacts);
  pile.add(new InfiniteCylinderDistance(R), U3{1, 20, 20}, -1._F, 0, Mesh(),
           0._F, 1, 0, 0.1, F3{1, 1, 1}, F3{0, 0, 0}, Q{0, 0, 0, 1}, Mesh());
  pile.build_grids(4 * kernel_radius);
  pile.reallocate_kinematics_on_device();
  cnst::set_num_boundaries(pile.get_size());
  cnst::set_contact_tolerance(0.05);

  // particles
  U num_particles_per_slice = 220;
  // U num_particles = 12 * kM * kQ*kQ;
  // U num_slices = 1;
  U num_slices = 2 * kM;
  U num_particles = 0;
  U max_num_particles = num_particles_per_slice * num_slices;
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
  U3 grid_res{kM, kQ, kQ};
  I3 grid_offset{-kM / 2, -kQ / 2, -kQ / 2};
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

  // store.map_graphical_pointers();
  // Runner::launch(max_num_particles, 256, [&](U grid_size, U block_size) {
  //   create_fluid_cylinder<F3, F><<<grid_size, block_size>>>(
  //       particle_x, max_num_particles, R - particle_radius * 2,
  //       num_particles_per_slice, particle_radius * 2._F,
  //       cylinder_length * -0.5_F);
  // });
  // store.unmap_graphical_pointers();
  // num_particles = max_num_particles - 2000;

  // // sample points
  // U num_sample_points = 21;
  // Variable<1, F3> sample_x = store.create<1, F3>({num_sample_points});
  // Variable<1, F3> sample_data = store.create<1, F3>({num_sample_points});
  // Variable<2, U> sample_neighbors =
  //     store.create<2, U>({num_sample_points,
  //     max_num_neighbors_per_particle});
  // Variable<1, U> sample_num_neighbors = store.create<1,
  // U>({num_sample_points});
  // // TODO: add boundary
  // std::vector<F3> sample_points_host(num_sample_points);
  // for (I i = 0; i < num_sample_points; ++i) {
  //   sample_points_host[i] = F3{0._F, 0._F,
  //                              R * 2._F / (num_sample_points + 1) *
  //                                  (i - static_cast<I>(num_sample_points) /
  //                                  2)};
  // }
  // sample_x.set_bytes(sample_points_host.data());
  // std::vector<F3> sample_data_host(num_sample_points);

  U frame_id = 0;
  F t = 0;
  F t_next_emission = 0;
  bool should_close = false;
  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, [&](ShadingProgram& program, Display& display) {
        // std::cout << "============= frame_id = " << frame_id << std::endl;
        if (should_close) {
          return;
        }

        store.map_graphical_pointers();
        // if (frame_id == 10000) {
        //   pid_length.set_zero();
        //   Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
        //     update_particle_grid<<<grid_size, block_size>>>(
        //         particle_x, pid, pid_length, num_particles);
        //   });
        //   Runner::launch(num_sample_points, 256, [&](U grid_size, U
        //   block_size) {
        //     make_neighbor_list_wrapped<<<grid_size, block_size>>>(
        //         sample_x, particle_x, pid, pid_length, sample_neighbors,
        //         sample_num_neighbors, num_sample_points);
        //   });

        //   Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
        //     compute_density_fluid_wrapped<<<grid_size, block_size>>>(
        //         particle_x, particle_neighbors, particle_num_neighbors,
        //         particle_density, num_particles);
        //   });
        //   Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
        //     compute_density_boundary<<<grid_size, block_size>>>(
        //         particle_x, particle_density, particle_boundary_xj,
        //         particle_boundary_volume, num_particles);
        //   });

        //   Runner::launch(num_sample_points, 256, [&](U grid_size, U
        //   block_size) {
        //     sample_fluid<<<grid_size, block_size>>>(
        //         sample_x, particle_x, particle_density, particle_v,
        //         sample_neighbors, sample_num_neighbors, sample_data,
        //         num_sample_points);
        //   });
        //   should_close = true;
        // }
        // start of simulation loop
        // F num_emission = 50;
        // if (frame_id > 5000 && frame_id % 300 == 1 && num_particles +
        // num_emission <= max_num_particles) {
        //   Runner::launch(num_emission, 256, [&](U grid_size, U block_size) {
        //     emit_cylinder<<<grid_size, block_size>>>(
        //         particle_x, particle_v, num_emission, num_particles, 2.0_F,
        //         F3{0._F, R*3._F/4._F, 0._F}, F3{0._F, -0.1_F, 0._F});
        //   });
        //   num_particles += num_emission;
        // }
        std::cout << "num_particles = " << num_particles << std::endl;
        std::vector<F3> particle_x_host(1);
        particle_x.get_bytes(particle_x_host.data(), sizeof(F3));
        F3 first_x = particle_x_host[0];
        std::cout << "x = " << first_x.x << ", " << first_x.y << ", "
                  << first_x.z << std::endl;
        for (U frame_interstep = 0; frame_interstep < 1; ++frame_interstep) {
          // emission
          F num_emission = 20;
          F3 emission_velocity = F3{0._F, -4.0_F, 0._F};
          F emission_speed = length(emission_velocity);
          F emission_interval = particle_radius * 2._F / emission_speed;
          if (t >= t_next_emission &&
              num_particles + num_emission <= max_num_particles) {
            Runner::launch(num_emission, 256, [&](U grid_size, U block_size) {
              emit_cylinder<<<grid_size, block_size>>>(
                  particle_x, particle_v, num_emission, num_particles, 2.0_F,
                  F3{0._F, 0._F, 0._F}, emission_velocity);
            });
            num_particles += num_emission;
            t_next_emission = t + emission_interval;
          }
          //
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
          F cfl_length_scale = particle_radius * 2._F;
          F cfl_factor = 0.04_F;
          F max_particle_speed =
              sqrt(Runner::max(particle_cfl_v2, num_particles));
          dt = cfl_factor * (cfl_length_scale / max_particle_speed);
          dt = min(5e-3, dt);
          std::cout << "dt = " << dt << " max speed = " << max_particle_speed
                    << std::endl;
          // if (max_particle_speed > 100.0 && frame_id > 1000) {
          //   should_close = true;
          // }

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
        }
        store.unmap_graphical_pointers();
        frame_id += 1;
        t += dt;
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
        glUniform3f(program.get_uniform_location("material.diffuse"), 0.2, 0.3,
                    0.87);
        glUniform3f(program.get_uniform_location("material.specular"), 0.8, 0.9,
                    0.9);
        glUniform1f(program.get_uniform_location("material.shininess"), 5.0);

        glBindBuffer(GL_ARRAY_BUFFER, particle_x.vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        for (I i = -1; i <= 1; ++i) {
          float wrap_length = grid_res.x * kernel_radius;
          glUniformMatrix4fv(
              program.get_uniform_location("M"), 1, GL_FALSE,
              glm::value_ptr(glm::translate(glm::mat4(1),
                                            glm::vec3{wrap_length * i, 0, 0})));
          glDrawArrays(GL_POINTS, 0, num_particles);
        }
        glDisableVertexAttribArray(0);
      }));

  // rigid mesh shader
  // https://github.com/opengl-tutorials/ogl
#include "alluvion/glsl/mesh_with_normal.frag"
#include "alluvion/glsl/mesh_with_normal.vert"
  display->add_shading_program(new ShadingProgram(
      kMeshWithNormalVertexShaderStr, kMeshWithNormalFragmentShaderStr,
      {"MVP", "V", "M", "camera_worldspace", "material.diffuse",
       "material.specular", "material.shininess", "directional_light.direction",
       "directional_light.ambient", "directional_light.diffuse",
       "directional_light.specular", "point_lights[0].position",
       "point_lights[0].constant", "point_lights[0].linear",
       "point_lights[0].quadratic", "point_lights[0].ambient",
       "point_lights[0].diffuse", "point_lights[0].specular",
       //
       "point_lights[1].position", "point_lights[1].constant",
       "point_lights[1].linear", "point_lights[1].quadratic",
       "point_lights[1].ambient", "point_lights[1].diffuse",
       "point_lights[1].specular"},
      [&pile](ShadingProgram& program, Display& display) {
        glm::mat4 const& projection_matrix =
            display.camera_.getProjectionMatrix();
        glm::mat4 const& view_matrix = display.camera_.getViewMatrix();
        glUniformMatrix4fv(program.get_uniform_location("V"), 1, GL_FALSE,
                           glm::value_ptr(view_matrix));
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

        for (U i = 0; i < pile.get_size(); ++i) {
          MeshBuffer const& mesh_buffer = pile.mesh_buffer_list_[i];
          if (mesh_buffer.vertex != 0 && mesh_buffer.index != 0) {
            glm::mat4 model_matrix = pile.get_matrix(i);
            glm::mat4 mvp_matrix =
                projection_matrix * view_matrix * model_matrix;

            glUniformMatrix4fv(program.get_uniform_location("M"), 1, GL_FALSE,
                               glm::value_ptr(model_matrix));
            glUniformMatrix4fv(program.get_uniform_location("MVP"), 1, GL_FALSE,
                               glm::value_ptr(mvp_matrix));
            glUniform3f(program.get_uniform_location("material.diffuse"), 0.2,
                        0.3, 0.87);
            glUniform3f(program.get_uniform_location("material.specular"), 0.8,
                        0.9, 0.9);
            glUniform1f(program.get_uniform_location("material.shininess"),
                        5.0);

            glBindBuffer(GL_ARRAY_BUFFER, mesh_buffer.vertex);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

            glBindBuffer(GL_ARRAY_BUFFER, mesh_buffer.normal);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_buffer.index);
            glDrawElements(GL_TRIANGLES, mesh_buffer.num_indices,
                           GL_UNSIGNED_INT, 0);
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
          }
        }
      }));
  display->run();
  // }}}
}
