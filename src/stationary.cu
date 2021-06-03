#include <glm/gtc/type_ptr.hpp>
#include <iostream>

#include "alluvion/colormaps.hpp"
#include "alluvion/constants.hpp"
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
  F particle_mass = 0.1;
  F dt = 2e-3;
  F3 gravity = {0._F, -9.81_F, 0._F};
  cnst::set_cubic_discretization_constants();
  cnst::set_kernel_radius(kernel_radius);
  cnst::set_particle_attr(particle_radius, particle_mass, density0);
  cnst::set_gravity(gravity);
  cnst::set_advanced_fluid_attr(0.1, 0.01, 0.1, 0.5, 0.05, 0.01);

  // rigids
  U max_num_contacts = 512;
  Pile pile(store, max_num_contacts);
  Mesh cube_mesh;
  cube_mesh.set_obj("cube.obj");
  Mesh sphere_mesh;
  F sphere_radius = 2.5_F;
  sphere_mesh.set_uv_sphere(sphere_radius, 24, 24);
  pile.add(cube_mesh, U3{20, 20, 20}, -1._F, 0, cube_mesh, 0._F, 1, 0, 0.1,
           F3{1, 1, 1}, F3{0, 0, 0}, Q{0, 0, 0, 1}, Mesh());
  pile.add(new SphereDistance(sphere_radius), U3{50, 50, 50}, 1._F, 0,
           sphere_mesh, 65.45_F, 1, 0, 0.2, F3{1, 1, 1}, F3{-5, -5, -5},
           Q{0, 0, 0, 1}, sphere_mesh);
  pile.build_grids(4 * kernel_radius);
  // pile.build_grids(0.1_F);
  pile.reallocate_kinematics_on_device();
  cnst::set_num_boundaries(pile.get_size());
  cnst::set_contact_tolerance(0.05);

  // particles
  U num_particles = 10000;
  GraphicalVariable<1, F3> particle_x =
      store.create_graphical<1, F3>({num_particles});
  GraphicalVariable<1, F> particle_normalized_attr =
      store.create_graphical<1, F>({num_particles});
  Variable<1, F3> particle_v = store.create<1, F3>({num_particles});
  Variable<1, F3> particle_a = store.create<1, F3>({num_particles});
  Variable<1, F> particle_density = store.create<1, F>({num_particles});
  Variable<1, F> particle_pressure = store.create<1, F>({num_particles});
  Variable<1, F> particle_last_pressure = store.create<1, F>({num_particles});
  Variable<2, F3> particle_boundary_xj =
      store.create<2, F3>({pile.get_size(), num_particles});
  Variable<2, F> particle_boundary_volume =
      store.create<2, F>({pile.get_size(), num_particles});
  Variable<2, F3> particle_force =
      store.create<2, F3>({pile.get_size(), num_particles});
  Variable<2, F3> particle_torque =
      store.create<2, F3>({pile.get_size(), num_particles});
  Variable<1, F> particle_aii = store.create<1, F>({num_particles});
  Variable<1, F3> particle_dii = store.create<1, F3>({num_particles});
  Variable<1, F3> particle_dij_pj = store.create<1, F3>({num_particles});
  Variable<1, F> particle_sum_tmp = store.create<1, F>({num_particles});
  Variable<1, F> particle_adv_density = store.create<1, F>({num_particles});
  Variable<1, F3> particle_pressure_accel =
      store.create<1, F3>({num_particles});

  // grid
  U3 grid_res{128, 128, 128};
  I3 grid_offset{-64, -64, -64};
  U max_num_particles_per_cell = 128;
  U max_num_neighbors_per_particle = 128;
  const F kCellWidthRelativeToKernelRadius =
      pow((sqrt(5.0) - 1.0) * 0.5, 1.0 / 3.0);
  cnst::init_grid_constants(grid_res, grid_offset);
  cnst::set_cell_width(kernel_radius * kCellWidthRelativeToKernelRadius);
  cnst::set_search_range(2.0 / kCellWidthRelativeToKernelRadius);
  cnst::set_max_num_particles_per_cell(max_num_particles_per_cell);
  cnst::set_max_num_neighbors_per_particle(max_num_neighbors_per_particle);
  Variable<4, U> pid = store.create<4, U>(
      {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell});
  Variable<3, U> pid_length =
      store.create<3, U>({grid_res.x, grid_res.y, grid_res.z});
  // neighbor
  Variable<2, U> particle_neighbors =
      store.create<2, U>({num_particles, max_num_neighbors_per_particle});
  Variable<1, U> particle_num_neighbors = store.create<1, U>({num_particles});

  store.map_graphical_pointers();
  Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
    create_fluid_block<F3, F>
        <<<grid_size, block_size>>>(particle_x, num_particles, 0, 0,
                                    F3{-6.0, -2.0, -6.0}, F3{6.0, 7.0, 6.0});
  });

  store.unmap_graphical_pointers();

  GLuint colormap_tex =
      display->create_colormap(kViridisData.data(), kViridisData.size());

  U frame_id = 0;
  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, {}, [&](ShadingProgram& program, Display& display) {
        // std::cout << "============= frame_id = " << frame_id << std::endl;

        store.map_graphical_pointers();
        // start of simulation loop
        for (U frame_interstep = 0; frame_interstep < 10; ++frame_interstep) {
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
              compute_particle_boundary<<<grid_size, block_size>>>(
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
            make_neighbor_list<<<grid_size, block_size>>>(
                particle_x, particle_x, pid, pid_length, particle_neighbors,
                particle_num_neighbors, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            compute_density<<<grid_size, block_size>>>(
                particle_x, particle_neighbors, particle_num_neighbors,
                particle_density, particle_boundary_xj,
                particle_boundary_volume, num_particles);
          });
          // compute_normal
          // compute_surface_tension_fluid
          // compute_surface_tension_boundary

          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            compute_viscosity_fluid<<<grid_size, block_size>>>(
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
          // update_dt

          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            predict_advection0_fluid_advect<<<grid_size, block_size>>>(
                particle_v, particle_a, dt, num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            predict_advection0_fluid<<<grid_size, block_size>>>(
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
            predict_advection1_fluid<<<grid_size, block_size>>>(
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

          for (U p_solve_iteration = 0; p_solve_iteration < 2;
               ++p_solve_iteration) {
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              pressure_solve_iteration0<<<grid_size, block_size>>>(
                  particle_x, particle_density, particle_last_pressure,
                  particle_dij_pj, particle_neighbors, particle_num_neighbors,
                  num_particles);
            });
            Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
              pressure_solve_iteration1_fluid<<<grid_size, block_size>>>(
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
            compute_pressure_accels_fluid<<<grid_size, block_size>>>(
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
            kinematic_integration<<<grid_size, block_size>>>(
                particle_x, particle_v, particle_pressure_accel, dt,
                num_particles);
          });
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            normalize_vector_magnitude<<<grid_size, block_size>>>(
                particle_v, particle_normalized_attr, 0._F, 2.0_F,
                num_particles);
          });

          // rigids
          for (U i = 0; i < pile.get_size(); ++i) {
            if (pile.mass_[i] == 0._F) continue;
            pile.force_[i] =
                Runner::sum(particle_force, num_particles, i * num_particles);
            pile.torque_[i] =
                Runner::sum(particle_torque, num_particles, i * num_particles);
          }

          // apply total force by fluid
          for (U i = 0; i < pile.get_size(); ++i) {
            if (pile.mass_[i] == 0._F) continue;
            pile.v_(i) += 1._F / pile.mass_[i] * pile.force_[i] * dt;
            pile.omega_(i) +=
                calculate_angular_acceleration(pile.inertia_tensor_[i],
                                               pile.q_[i], pile.torque_[i]) *
                dt;
            // clear accleration
            pile.a_[i] = gravity;
          }
          // semi_implicit_euler
          for (U i = 0; i < pile.get_size(); ++i) {
            if (pile.mass_[i] == 0._F) continue;
            pile.q_[i] += dt * calculate_dq(pile.omega_(i), pile.q_[i]);
            pile.q_[i] = normalize(pile.q_[i]);

            // pile.x_(i) += (pile.a_[i] * dt + pile.v_(i)) * dt;
            // pile.v_(i) += pile.a_[i] * dt;
            F3 dx = (pile.a_[i] * dt + pile.v_(i)) * dt;
            pile.x_(i) += dx;
            pile.v_(i) = 1 / dt * dx;
          }
          pile.find_contacts();
          pile.solve_contacts();
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
       "camera_worldspace", "material.specular", "material.shininess",
       "directional_light.direction", "directional_light.ambient",
       "directional_light.diffuse", "directional_light.specular",
       "point_lights[0].position", "point_lights[0].constant",
       "point_lights[0].linear", "point_lights[0].quadratic",
       "point_lights[0].ambient", "point_lights[0].diffuse",
       "point_lights[0].specular",
       //
       "point_lights[1].position", "point_lights[1].constant",
       "point_lights[1].linear", "point_lights[1].quadratic",
       "point_lights[1].ambient", "point_lights[1].diffuse",
       "point_lights[1].specular"

      },
      {std::make_tuple(particle_x.vbo_, 3, 0),
       std::make_tuple(particle_normalized_attr.vbo_, 1, 0)},
      [&](ShadingProgram& program, Display& display) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUniformMatrix4fv(program.get_uniform_location("M"), 1, GL_FALSE,
                           glm::value_ptr(glm::mat4(1)));
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
        glUniform3f(program.get_uniform_location("material.specular"), 0.8, 0.9,
                    0.9);
        glUniform1f(program.get_uniform_location("material.shininess"), 5.0);

        glBindTexture(GL_TEXTURE_1D, colormap_tex);
        glDrawArrays(GL_POINTS, 0, num_particles);
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
      {std::make_tuple(0, 3, 0), std::make_tuple(0, 3, 0)},
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
