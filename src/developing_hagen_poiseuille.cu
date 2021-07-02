#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "alluvion/colormaps.hpp"
#include "alluvion/constants.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/solver_df.hpp"
#include "alluvion/store.hpp"
#include "alluvion/typesetter.hpp"

using namespace alluvion;
using namespace alluvion::dg;

int main(void) {
  I kQ = 5;
  std::string particle_x_filename(
      "/home/kennychufk/workspace/cppWs/alluvion/x5-5-7214-rest.alu");
  F pressure_gradient_acc_x = 1.28e-4;
  F viscosity = 4.04909890e-06;
  F boundary_viscosity = 5.07019036e-06;
  F dt = 1e-2;
  std::vector<F> sample_ts = {320.0};

  Store store;
  Display* display = store.create_display(1920, 1080, "particle view", false);
  Runner<F> runner;
  GLuint screenshot_fbo = display->create_framebuffer();

  F particle_radius = 0.0025_F;
  F kernel_radius = particle_radius * 4.0_F;
  F density0 = 1000.0_F;
  F cubical_particle_volume =
      8 * particle_radius * particle_radius * particle_radius;
  F volume_relative_to_cube = 0.8_F;
  F particle_mass =
      cubical_particle_volume * volume_relative_to_cube * density0;

  F3 pressure_gradient_acc = F3{0._F, pressure_gradient_acc_x, 0._F};

  store.get_cn<F>().set_cubic_discretization_constants();
  store.get_cn<F>().set_kernel_radius(kernel_radius);
  store.get_cn<F>().set_particle_attr(particle_radius, particle_mass, density0);
  store.get_cn<F>().gravity = pressure_gradient_acc;
  store.get_cn<F>().boundary_epsilon = 1e-9_F;
  F vorticity = 0.01_F;
  F inertia_inverse = 0.1_F;
  F viscosity_omega = 0.5_F;
  F surface_tension_coeff = 0.05_F;
  F surface_tension_boundary_coeff = 0.01_F;
  store.get_cn<F>().viscosity = viscosity;

  I kM = 5;
  F cylinder_length = 2._F * kM * kernel_radius;
  F R = kernel_radius * kQ;

  // rigids
  F restitution = 1._F;
  F friction = 0._F;
  U max_num_contacts = 512;
  Pile<F3, Q, F> pile(store, max_num_contacts);
  pile.add(new InfiniteCylinderDistance<F3, F>(R), U3{64, 1, 64}, -1._F, 0,
           Mesh(), 0._F, restitution, friction, boundary_viscosity, F3{1, 1, 1},
           F3{0, 0, 0}, Q{0, 0, 0, 1}, Mesh());
  pile.build_grids(4 * kernel_radius);
  pile.reallocate_kinematics_on_device();
  store.get_cn<F>().contact_tolerance = particle_radius;

  // particles
  U max_num_particles = static_cast<U>(kPi<F> * R * R * cylinder_length *
                                       density0 / particle_mass);

  // grid
  U3 grid_res{static_cast<U>(kQ * 2), static_cast<U>(kM * 2),
              static_cast<U>(kQ * 2)};
  I3 grid_offset{-kQ, -kM, -kQ};
  U max_num_particles_per_cell = 64;
  U max_num_neighbors_per_particle = 64;
  store.get_cni().grid_res = grid_res;
  store.get_cni().grid_offset = grid_offset;
  store.get_cni().max_num_particles_per_cell = max_num_particles_per_cell;
  store.get_cni().max_num_neighbors_per_particle =
      max_num_neighbors_per_particle;
  store.get_cn<F>().set_wrap_length(grid_res.y * kernel_radius);

  std::unique_ptr<GraphicalVariable<1, F3>> particle_x(
      store.create_graphical<1, F3>({max_num_particles}));
  std::unique_ptr<GraphicalVariable<1, F>> particle_normalized_attr(
      store.create_graphical<1, F>({max_num_particles}));
  Variable<1, F3> particle_v = store.create<1, F3>({max_num_particles});
  Variable<1, F3> particle_a = store.create<1, F3>({max_num_particles});
  Variable<1, F> particle_density = store.create<1, F>({max_num_particles});
  Variable<2, F3> particle_boundary_xj =
      store.create<2, F3>({pile.get_size(), max_num_particles});
  Variable<2, F> particle_boundary_volume =
      store.create<2, F>({pile.get_size(), max_num_particles});
  Variable<2, F3> particle_force =
      store.create<2, F3>({pile.get_size(), max_num_particles});
  Variable<2, F3> particle_torque =
      store.create<2, F3>({pile.get_size(), max_num_particles});
  Variable<1, F> particle_cfl_v2 = store.create<1, F>({max_num_particles});
  Variable<1, F> particle_dfsph_factor =
      store.create<1, F>({max_num_particles});
  Variable<1, F> particle_kappa = store.create<1, F>({max_num_particles});
  Variable<1, F> particle_kappa_v = store.create<1, F>({max_num_particles});
  Variable<1, F> particle_density_adv = store.create<1, F>({max_num_particles});

  Variable<4, Q> pid = store.create<4, Q>(
      {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell});
  Variable<3, U> pid_length =
      store.create<3, U>({grid_res.x, grid_res.y, grid_res.z});
  Variable<2, Q> particle_neighbors =
      store.create<2, Q>({max_num_particles, max_num_neighbors_per_particle});
  Variable<1, U> particle_num_neighbors =
      store.create<1, U>({max_num_particles});

  SolverDf<F3, Q, F> solver_df(
      runner, pile, *particle_x, *particle_normalized_attr, particle_v,
      particle_a, particle_density, particle_boundary_xj,
      particle_boundary_volume, particle_force, particle_torque,
      particle_cfl_v2, particle_dfsph_factor, particle_kappa, particle_kappa_v,
      particle_density_adv, pid, pid_length, particle_neighbors,
      particle_num_neighbors);
  solver_df.dt = dt;
  solver_df.max_dt = 0.001_F;
  solver_df.min_dt = 0.0001_F;
  solver_df.cfl = 0.4_F;
  solver_df.particle_radius = particle_radius;

  // sample points
  U num_sample_planes = 14;
  U num_samples_per_plane = 31;
  U num_samples = num_samples_per_plane * num_sample_planes;
  Variable<1, F3> sample_x = store.create<1, F3>({num_samples});
  Variable<1, F3> sample_data3 = store.create<1, F3>({num_samples});
  Variable<2, Q> sample_neighbors =
      store.create<2, Q>({num_samples, max_num_neighbors_per_particle});
  Variable<1, U> sample_num_neighbors = store.create<1, U>({num_samples});
  Variable<2, F3> sample_boundary_xj =
      store.create<2, F3>({pile.get_size(), num_samples});
  Variable<2, F> sample_boundary_volume =
      store.create<2, F>({pile.get_size(), num_samples});
  {
    std::vector<F3> sample_x_host(num_samples);
    F distance_between_sample_planes = cylinder_length / num_sample_planes;
    for (I i = 0; i < num_samples; ++i) {
      I plane_id = i / num_samples_per_plane;
      I id_in_plane = i % num_samples_per_plane;
      sample_x_host[i] = F3{
          R * 2._F / (num_samples_per_plane + 1) *
              (id_in_plane - static_cast<I>(num_samples_per_plane) / 2),
          cylinder_length * -0.5_F + distance_between_sample_planes * plane_id,
          0._F};
    }
    sample_x.set_bytes(sample_x_host.data());
  }
  std::vector<F3> sample_data3_host(num_samples);
  std::vector<F> vx(num_samples_per_plane * sample_ts.size());

  F t = 0;

  I sampling_cursor = 0;

  store.copy_cn<F>();
  store.map_graphical_pointers();
  solver_df.num_particles = particle_x->read_file(particle_x_filename.c_str());
  store.unmap_graphical_pointers();
  bool should_close = false;
  int freeze_beginning = 0;

  const char* font_filename = "/usr/share/fonts/truetype/freefont/FreeMono.ttf";
  Typesetter typesetter(display, font_filename, 0, 48);
  typesetter.load_ascii();

  display->camera_.setEye(0._F, 0._F, R * 3._F);
  display->camera_.setUp(1._F, 0._F, 0._F);
  display->camera_.setClipPlanes(particle_radius * 10._F, R * 20._F);
  display->camera_.update();
  display->update_trackball_camera();

  GLuint colormap_tex =
      display->create_colormap(kViridisData.data(), kViridisData.size());

  GLuint glyph_quad = display->create_dynamic_array_buffer<float4>(6, nullptr);
  constexpr float kScreenQuadXYTex[] = {// positions   // texCoords
                                        -1.0f, 1.0f, 0.0f, 1.0f,  -1.0f, -1.0f,
                                        0.0f,  0.0f, 1.0f, -1.0f, 1.0f,  0.0f,
                                        -1.0f, 1.0f, 0.0f, 1.0f,  1.0f,  -1.0f,
                                        1.0f,  0.0f, 1.0f, 1.0f,  1.0f,  1.0f};
  GLuint screen_quad =
      display->create_dynamic_array_buffer<float4>(6, kScreenQuadXYTex);

  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, {}, [&](ShadingProgram& program, Display& display) {
        if (should_close) {
          return;
        }
        if (++freeze_beginning < 400) {
          return;
        }
        store.map_graphical_pointers();

        for (int subframe_interval = 0; subframe_interval < 10;
             ++subframe_interval) {
          if (t >= sample_ts[sampling_cursor]) {
            should_close = true;
          }
          solver_df.step<1, 0>();
          t += solver_df.dt;
        }
        solver_df.colorize_speed(0, 0.004);
        pid_length.set_zero();
        Runner<F>::launch(
            solver_df.num_particles, 256, [&](U grid_size, U block_size) {
              update_particle_grid<<<grid_size, block_size>>>(
                  *particle_x, pid, pid_length, solver_df.num_particles);
            });
        Runner<F>::launch(num_samples, 256, [&](U grid_size, U block_size) {
          make_neighbor_list<1><<<grid_size, block_size>>>(
              sample_x, pid, pid_length, sample_neighbors, sample_num_neighbors,
              num_samples);
        });
        Runner<F>::launch(
            solver_df.num_particles, 256, [&](U grid_size, U block_size) {
              compute_density<<<grid_size, block_size>>>(
                  *particle_x, particle_neighbors, particle_num_neighbors,
                  particle_density, particle_boundary_xj,
                  particle_boundary_volume, solver_df.num_particles);
            });
        Runner<F>::launch(num_samples, 256, [&](U grid_size, U block_size) {
          sample_fluid<<<grid_size, block_size>>>(
              sample_x, *particle_x, particle_density, particle_v,
              sample_neighbors, sample_num_neighbors, sample_data3,
              num_samples);
        });
        for (F& value : vx) {
          value = 0._F;
        }
        sample_data3.get_bytes(sample_data3_host.data());
        for (I i = 0; i < sample_data3_host.size(); ++i) {
          vx[i % num_samples_per_plane] +=
              sample_data3_host[i].y / num_sample_planes;
        }
        store.unmap_graphical_pointers();
      }));

#include "alluvion/glsl/particle.frag"
#include "alluvion/glsl/particle.vert"
  display->add_shading_program(new ShadingProgram(
      kParticleVertexShaderStr.c_str(), kParticleFragmentShaderStr.c_str(),
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
      {std::make_tuple(particle_x->vbo_, 3, GL_F, 0),
       std::make_tuple(particle_normalized_attr->vbo_, 1, GL_F, 0)},
      [&](ShadingProgram& program, Display& display) {
        glBindFramebuffer(GL_FRAMEBUFFER,
                          display.get_framebuffer(screenshot_fbo).fbo_);
        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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
                    0.15f, 0.15f, 0.15f);
        glUniform3f(program.get_uniform_location("directional_light.diffuse"),
                    0.5f, 0.5f, 0.5f);
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
        glUniform3f(program.get_uniform_location("material.specular"), 0.04f,
                    0.04f, 0.04f);
        glUniform1f(program.get_uniform_location("material.shininess"), 5.0f);

        glBindTexture(GL_TEXTURE_1D, colormap_tex);
        for (I i = 0; i <= 0; ++i) {
          float wrap_length = grid_res.y * kernel_radius;
          glUniformMatrix4fv(
              program.get_uniform_location("M"), 1, GL_FALSE,
              glm::value_ptr(glm::translate(glm::mat4(1),
                                            glm::vec3{wrap_length * i, 0, 0})));
          glDrawArrays(GL_POINTS, 0, solver_df.num_particles);
        }
      }));

#include "alluvion/glsl/glyph.frag"
#include "alluvion/glsl/glyph.vert"
  display->add_shading_program(new ShadingProgram(
      kGlyphVertexShaderStr.c_str(), kGlyphFragmentShaderStr.c_str(),
      {
          "projection",
          "text_color",
      },
      {std::make_tuple(glyph_quad, 4, GL_FLOAT, 0)},
      [&](ShadingProgram& program, Display& display) {
        glm::mat4 projection =
            glm::ortho(0.0f, static_cast<GLfloat>(display.width_), 0.0f,
                       static_cast<GLfloat>(display.height_));
        glUniformMatrix4fv(program.get_uniform_location("projection"), 1,
                           GL_FALSE, glm::value_ptr(projection));
        glUniform3f(program.get_uniform_location("text_color"), 1.0f, 1.0f,
                    1.0f);

        std::stringstream time_text;
        time_text << "t = " << std::fixed << std::setprecision(2)
                  << std::setw(6) << t
                  << "s  central speed = " << std::scientific
                  << std::setprecision(4) << std::setw(6) << vx[15] << "m/s";
        std::string text = time_text.str();

        typesetter.start(display.width_ * 0.4f, display.height_ * 0.05f, 1.0f);
        for (std::string::const_iterator c = text.begin(); c != text.end();
             c++) {
          glBindTexture(GL_TEXTURE_2D, typesetter.place_glyph(*c));
          glBindBuffer(GL_ARRAY_BUFFER, glyph_quad);
          glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(typesetter.vertices_info_),
                          typesetter.vertices_info_);
          glBindBuffer(GL_ARRAY_BUFFER, 0);
          glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
      }));
#include "alluvion/glsl/screen.frag"
#include "alluvion/glsl/screen.vert"
  display->add_shading_program(new ShadingProgram(
      kScreenVertexShaderStr.c_str(), kScreenFragmentShaderStr.c_str(), {},
      {std::make_tuple(screen_quad, 4, GL_FLOAT, 0)},
      [&](ShadingProgram& program, Display& display) {
        // display.get_framebuffer(screenshot_fbo).write("test.bmp");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D,
                      display.get_framebuffer(screenshot_fbo).color_tex_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
      }));
  display->run();
  return 0;
}
