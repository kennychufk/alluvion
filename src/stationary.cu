#include <glm/gtc/type_ptr.hpp>
#include <iostream>

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
  F dt = 1e-3;
  F gravity = -9.81;
  cnst::set_cubic_discretization_constants();
  cnst::set_kernel_radius(kernel_radius);
  cnst::set_particle_attr(particle_radius, particle_mass, density0);
  cnst::set_gravity(gravity);

  U num_particles = 10000;
  U3 grid_res{128, 128, 128};
  U num_grid_cells = grid_res.x * grid_res.y * grid_res.z;
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

  // rigids
  Pile pile(store);
  pile.add("cube.obj", U3{50, 50, 50}, -1.0_F, 0, nullptr, 1, 1, 0, 0.2,
           F3{1, 1, 1}, F3{0, 0, 0}, Q{0, 0, 0, 1}, nullptr);
  pile.add(new SphereDistance(3.0_F), U3{50, 50, 50}, 1.0_F, 0, {}, 1, 1, 0,
           0.2, F3{1, 1, 1}, F3{-6, -6, -6}, Q{0, 0, 0, 1}, {}, {});
  pile.build_grids(4 * kernel_radius);
  pile.reallocate_kinematics_on_device();

  // particles
  GraphicalVariable<1, F3> particle_x =
      store.create_graphical<1, F3>({num_particles});
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
        <<<grid_size, block_size>>>(particle_x, num_particles, 0, 1,
                                    F3{-5.0, -5.0, -5.0}, F3{5.0, 5.0, 5.0});
  });

  store.unmap_graphical_pointers();

  U frame_id = 0;
  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, [&](ShadingProgram& program, Display& display) {
        // std::cout << "============= frame_id = " << frame_id << std::endl;

        store.map_graphical_pointers();
        // start of simulation loop
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
                particle_boundary_xj, particle_boundary_volume, num_particles);
          });
        });
        Runner::launch(num_grid_cells, 256, [&](U grid_size, U block_size) {
          clear_particle_grid<<<grid_size, block_size>>>(pid_length,
                                                         num_grid_cells);
        });
        Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
          update_particle_grid<<<grid_size, block_size>>>(
              particle_x, pid, pid_length, num_particles);
        });
        Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
          make_neighbor_list<<<grid_size, block_size>>>(
              particle_x, pid, pid_length, particle_neighbors,
              particle_num_neighbors, num_particles);
        });
        Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
          compute_density_fluid<<<grid_size, block_size>>>(
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
              particle_x, particle_density, particle_dii, particle_boundary_xj,
              particle_boundary_volume, num_particles);
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

        for (U p_solve_iteration = 0; p_solve_iteration < 5;
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
                particle_boundary_xj, particle_boundary_volume, num_particles);
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

        store.unmap_graphical_pointers();
        frame_id += 1;
      }));
  // {{{
  display->add_shading_program(new ShadingProgram(
      R"CODE(
#version 330 core
layout(location = 0) in vec3 x;
uniform mat4 view_matrix;
uniform mat4 clip_matrix;
uniform vec2 screen_dimension;
uniform float point_scale;

out vec3 eyePos;
out float eyeRadius;

void main() {
  vec4 camera_space_x4 = view_matrix * vec4(x, 1.0);

  eyePos = camera_space_x4.xyz;
  eyeRadius = point_scale / -camera_space_x4.z / screen_dimension.y;
  gl_Position = clip_matrix * camera_space_x4;
  gl_PointSize = point_scale / -camera_space_x4.z;
}
)CODE",
      R"CODE(
#version 330 core
uniform vec4 base_color;

out vec4 output_color;

void main() {
  const vec3 light_direction = vec3(0.577, 0.577, 0.577);

  vec3 N;
  N.xy = gl_PointCoord * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
  float N_squared = dot(N.xy, N.xy);
  if (N_squared > 1.0) discard;
  N.z = sqrt(1.0 - N_squared);

  float diffuse = max(0.0, dot(light_direction, N));
  output_color = base_color * diffuse;
}
)CODE",
      {"view_matrix", "clip_matrix", "screen_dimension", "point_scale",
       "base_color"},
      [&particle_x, num_particles](ShadingProgram& program, Display& display) {
        glm::mat4 clip_matrix = glm::perspective(
            glm::radians(45.0f),
            display.width_ / static_cast<GLfloat>(display.height_), .01f,
            100.f);

        glUniformMatrix4fv(program.get_uniform_location("view_matrix"), 1,
                           GL_FALSE,
                           glm::value_ptr(display.camera_.getMatrix()));
        glUniformMatrix4fv(program.get_uniform_location("clip_matrix"), 1,
                           GL_FALSE, glm::value_ptr(clip_matrix));
        glUniform2f(program.get_uniform_location("screen_dimension"),
                    static_cast<GLfloat>(display.width_),
                    static_cast<GLfloat>(display.height_));
        glUniform1f(program.get_uniform_location("point_scale"), 400);
        glUniform4f(program.get_uniform_location("base_color"), 1.0, 1.0, 1.0,
                    1.0);

        glBindBuffer(GL_ARRAY_BUFFER, particle_x.vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_POINTS, 0, num_particles);
        glDisableVertexAttribArray(0);
      }));

  // rigid mesh shader
  display->add_shading_program(new ShadingProgram(
      R"CODE(
#version 330 core
layout(location = 0) in vec3 x;
uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 clip_matrix;

void main() {
  gl_Position = clip_matrix * view_matrix * model_matrix * vec4(x, 1.0);
}
)CODE",
      R"CODE(
#version 330 core
uniform vec4 base_color;

out vec4 output_color;

void main() {
  output_color = base_color;
}
)CODE",
      {"model_matrix", "view_matrix", "clip_matrix", "base_color"},
      [&pile](ShadingProgram& program, Display& display) {
        glm::mat4 clip_matrix = glm::perspective(
            glm::radians(45.0f),
            display.width_ / static_cast<GLfloat>(display.height_), .01f,
            100.f);

        glm::mat4 model_matrix = pile.get_matrix(0);
        glUniformMatrix4fv(program.get_uniform_location("model_matrix"), 1,
                           GL_FALSE, glm::value_ptr(model_matrix));
        glUniformMatrix4fv(program.get_uniform_location("view_matrix"), 1,
                           GL_FALSE,
                           glm::value_ptr(display.camera_.getMatrix()));
        glUniformMatrix4fv(program.get_uniform_location("clip_matrix"), 1,
                           GL_FALSE, glm::value_ptr(clip_matrix));
        glUniform4f(program.get_uniform_location("base_color"), 0.9, 0.3, 0.4,
                    1.0);

        MeshBuffer const& mesh_buffer = pile.mesh_buffer_list_[0];
        glBindBuffer(GL_ARRAY_BUFFER, mesh_buffer.vertex);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_buffer.index);
        glDrawElements(GL_TRIANGLES, mesh_buffer.num_indices, GL_UNSIGNED_INT,
                       0);
        glDisableVertexAttribArray(0);
      }));
  display->run();
  // }}}
}
