#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <memory>

#include "alluvion/colormaps.hpp"
#include "alluvion/constants.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/solver_ii.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;
using namespace alluvion::dg;

int main(void) {
  Store store;
  Display* display = store.create_display(800, 600, "particle view");
  Runner runner;

  F particle_radius = 0.025;
  F kernel_radius = 0.1;
  F density0 = 1000.0;
  F particle_mass = 0.1;
  F dt = 2e-3;
  F3 gravity = {0._F, -9.81_F, 0._F};
  store.get_cn<F>().set_cubic_discretization_constants();
  store.get_cn<F>().set_kernel_radius(kernel_radius);
  store.get_cn<F>().set_particle_attr(particle_radius, particle_mass, density0);
  store.get_cn<F>().set_gravity(gravity);
  store.get_cn<F>().set_advanced_fluid_attr(0.001, 0.01, 0.1, 0.5, 0.05, 0.01);

  // rigids
  U max_num_contacts = 512;
  Pile<F3, Q, F> pile(store, max_num_contacts);
  Mesh cube_mesh;
  cube_mesh.set_obj("cube.obj");
  Mesh sphere_mesh;
  F sphere_radius = 0.1_F;
  sphere_mesh.set_uv_sphere(sphere_radius, 24, 24);
  pile.add(cube_mesh, U3{80, 60, 30}, -1._F, 0, cube_mesh, 0._F, 1, 0, 0.0,
           F3{1, 1, 1}, F3{0, 1.5, 0}, Q{0, 0, 0, 1}, Mesh());
  // pile.add(new SphereDistance<F3, F>(sphere_radius), U3{50, 50, 50}, 1._F, 0,
  //          sphere_mesh, 3.2_F, 0.4, 0, 0.2, F3{1, 1, 1}, F3{0, 0.4, -0},
  //          Q{0, 0, 0, 1}, sphere_mesh);
  pile.build_grids(4 * kernel_radius);
  // pile.build_grids(0.1_F);
  pile.reallocate_kinematics_on_device();
  pile.set_gravity(gravity);
  store.get_cni().set_num_boundaries(pile.get_size());
  store.get_cn<F>().set_contact_tolerance(0.05);

  // particles
  U num_particles = 6859;

  // grid
  U3 grid_res{128, 128, 128};
  I3 grid_offset{-64, -64, -64};
  U max_num_particles_per_cell = 64;
  U max_num_neighbors_per_particle = 64;
  store.get_cni().init_grid_constants(grid_res, grid_offset);
  store.get_cni().set_max_num_particles_per_cell(max_num_particles_per_cell);
  store.get_cni().set_max_num_neighbors_per_particle(
      max_num_neighbors_per_particle);

  std::unique_ptr<GraphicalVariable<1, F3>> particle_x(
      store.create_graphical<1, F3>({num_particles}));
  std::unique_ptr<GraphicalVariable<1, F>> particle_normalized_attr(
      store.create_graphical<1, F>({num_particles}));
  Variable<1, F3> particle_v = store.create<1, F3>({num_particles});
  Variable<1, F3> particle_a = store.create<1, F3>({num_particles});
  Variable<1, F> particle_density = store.create<1, F>({num_particles});
  Variable<2, F3> particle_boundary_xj =
      store.create<2, F3>({pile.get_size(), num_particles});
  Variable<2, F> particle_boundary_volume =
      store.create<2, F>({pile.get_size(), num_particles});
  Variable<2, F3> particle_force =
      store.create<2, F3>({pile.get_size(), num_particles});
  Variable<2, F3> particle_torque =
      store.create<2, F3>({pile.get_size(), num_particles});
  Variable<1, F> particle_cfl_v2 = store.create<1, F>({num_particles});
  Variable<1, F> particle_pressure = store.create<1, F>({num_particles});
  Variable<1, F> particle_last_pressure = store.create<1, F>({num_particles});
  Variable<1, F> particle_aii = store.create<1, F>({num_particles});
  Variable<1, F3> particle_dii = store.create<1, F3>({num_particles});
  Variable<1, F3> particle_dij_pj = store.create<1, F3>({num_particles});
  Variable<1, F> particle_sum_tmp = store.create<1, F>({num_particles});
  Variable<1, F> particle_adv_density = store.create<1, F>({num_particles});
  Variable<1, F3> particle_pressure_accel =
      store.create<1, F3>({num_particles});
  Variable<1, F> particle_density_err = store.create<1, F>({num_particles});
  Variable<4, Q> pid = store.create<4, Q>(
      {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell});
  Variable<3, U> pid_length =
      store.create<3, U>({grid_res.x, grid_res.y, grid_res.z});
  Variable<2, Q> particle_neighbors =
      store.create<2, Q>({num_particles, max_num_neighbors_per_particle});
  Variable<1, U> particle_num_neighbors = store.create<1, U>({num_particles});

  SolverIi<F3, Q, F> solver_ii(
      runner, pile, *particle_x, *particle_normalized_attr, particle_v,
      particle_a, particle_density, particle_boundary_xj,
      particle_boundary_volume, particle_force, particle_torque,
      particle_cfl_v2, particle_pressure, particle_last_pressure, particle_aii,
      particle_dii, particle_dij_pj, particle_sum_tmp, particle_adv_density,
      particle_pressure_accel, particle_density_err, pid, pid_length,
      particle_neighbors, particle_num_neighbors);
  solver_ii.num_particles = num_particles;
  solver_ii.dt = dt;
  solver_ii.max_dt = 0.005;
  solver_ii.min_dt = 0.0001;
  solver_ii.cfl = 0.4;
  solver_ii.particle_radius = particle_radius;

  store.copy_cn<F>();
  store.map_graphical_pointers();
  Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
    create_fluid_block<F3, F>
        <<<grid_size, block_size>>>(*particle_x, num_particles, 0, 0,
                                    F3{-1.95, -0.0, -0.5}, F3{-0.95, 1.0, 0.5});
  });

  store.unmap_graphical_pointers();

  display->camera_.setEye(0.f, 06.00f, 6.0f);
  display->camera_.setCenter(0.f, -0.20f, 0.f);
  display->camera_.update();
  display->update_trackball_camera();

  GLuint colormap_tex =
      display->create_colormap(kViridisData.data(), kViridisData.size());

  U frame_id = 0;
  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, {}, [&](ShadingProgram& program, Display& display) {
        // std::cout << "============= frame_id = " << frame_id << std::endl;

        store.map_graphical_pointers();
        // start of simulation loop
        for (U frame_interstep = 0; frame_interstep < 10; ++frame_interstep) {
          solver_ii.step<0>();
        }
        solver_ii.colorize_speed(0, 2);
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
      {std::make_tuple(particle_x->vbo_, 3, 0),
       std::make_tuple(particle_normalized_attr->vbo_, 1, 0)},
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
