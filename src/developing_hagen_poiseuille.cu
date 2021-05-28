#include <ft2build.h>

#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "alluvion/colormaps.hpp"
#include "alluvion/constants.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"
#include FT_FREETYPE_H

using namespace alluvion;
using namespace alluvion::dg;

struct Character {
  unsigned int TextureID;  // ID handle of the glyph texture
  glm::ivec2 Size;         // Size of glyph
  glm::ivec2 Bearing;      // Offset from baseline to left/top of glyph
  unsigned int Advance;    // Horizontal offset to advance to next glyph
};
std::map<GLchar, Character> Characters;

int main(void) {
  I kQ = 5;
  std::string particle_x_filename(
      "/home/kennychufk/workspace/cppWs/alluvion/x5-rest2.alu");
  F pressure_gradient_acc_x = 1.28e-4;
  F viscosity = 2.14928446e-05;
  F boundary_viscosity = 6.34569921e-05;
  F dt = 1e-2;
  std::vector<F> sample_ts = {320.0};

  Store store;
  Display* display = store.create_display(800, 600, "particle view");

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
  GraphicalVariable<1, F> particle_normalized_attr =
      store.create_graphical<1, F>({max_num_particles});
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
    F distance_between_sample_planes = cylinder_length / num_sample_planes;
    for (I i = 0; i < num_samples; ++i) {
      I plane_id = i / num_samples_per_plane;
      I id_in_plane = i % num_samples_per_plane;
      sample_x_host[i] = F3{
          cylinder_length * -0.5_F + distance_between_sample_planes * plane_id,
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

  store.map_graphical_pointers();
  num_particles = particle_x.read_file(particle_x_filename.c_str());
  store.unmap_graphical_pointers();
  bool should_close = false;
  int freeze_beginning = 0;

  FT_Library ft;
  // All functions return a value different than 0 whenever an error occurred
  if (FT_Init_FreeType(&ft)) {
    std::cout << "ERROR::FREETYPE: Could not init FreeType Library"
              << std::endl;
    return -1;
  }

  // find path to font
  std::string font_name = "/usr/share/fonts/truetype/freefont/FreeMono.ttf";
  if (font_name.empty()) {
    std::cout << "ERROR::FREETYPE: Failed to load font_name" << std::endl;
    return -1;
  }

  // load font as face
  FT_Face face;
  if (FT_New_Face(ft, font_name.c_str(), 0, &face)) {
    std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
    return -1;
  } else {
    // set size to load glyphs as
    FT_Set_Pixel_Sizes(face, 0, 48);

    // disable byte-alignment restriction
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // load first 128 characters of ASCII set
    for (unsigned char c = 0; c < 128; c++) {
      // Load character glyph
      if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
        std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
        continue;
      }
      // generate texture
      unsigned int texture;
      glGenTextures(1, &texture);
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width,
                   face->glyph->bitmap.rows, 0, GL_RED, GL_UNSIGNED_BYTE,
                   face->glyph->bitmap.buffer);
      // set texture options
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      // now store character for later use
      Character character = {
          texture,
          glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
          glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
          static_cast<unsigned int>(face->glyph->advance.x)};
      Characters.insert(std::pair<char, Character>(c, character));
    }
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  // destroy FreeType once we're finished
  FT_Done_Face(face);
  FT_Done_FreeType(ft);

  display->camera_.setEye(0._F, 0._F, R * 3._F);
  display->camera_.setUp(1._F, 0._F, 0._F);
  display->camera_.setClipPlanes(particle_radius * 10._F, R * 20._F);
  display->update_trackball_camera();

  GLuint colormap_tex =
      display->create_colormap(kViridisData.data(), kViridisData.size());

  GLuint glyph_vao;
  glGenVertexArrays(1, &glyph_vao);
  GLuint glyph_quad =
      GraphicalAllocator::allocate_dynamic_array_buffer<float4>(6, nullptr);

  glBindVertexArray(glyph_vao);
  glBindBuffer(GL_ARRAY_BUFFER, glyph_quad);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, [&](ShadingProgram& program, Display& display) {
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
          Runner::launch(num_particles, 256, [&](U grid_size, U block_size) {
            normalize_vector_magnitude<<<grid_size, block_size>>>(
                particle_v, particle_normalized_attr, 0._F, 0.004_F,
                num_particles);
          });
          t += dt;
          step_id += 1;
        }
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
              sample_x, particle_x, particle_density, particle_v,
              sample_neighbors, sample_num_neighbors, sample_data3,
              num_samples);
        });
        for (F& value : vx) {
          value = 0._F;
        }
        sample_data3.get_bytes(sample_data3_host.data());
        for (I i = 0; i < sample_data3_host.size(); ++i) {
          vx[i % num_samples_per_plane] +=
              sample_data3_host[i].x / num_sample_planes;
        }
        store.unmap_graphical_pointers();
      }));

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
      [&](ShadingProgram& program, Display& display) {
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
        glBindBuffer(GL_ARRAY_BUFFER, particle_x.vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, particle_normalized_attr.vbo_);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
        for (I i = 0; i <= 0; ++i) {
          float wrap_length = grid_res.x * kernel_radius;
          glUniformMatrix4fv(
              program.get_uniform_location("M"), 1, GL_FALSE,
              glm::value_ptr(glm::translate(glm::mat4(1),
                                            glm::vec3{wrap_length * i, 0, 0})));
          glDrawArrays(GL_POINTS, 0, num_particles);
        }
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
      }));

#include "alluvion/glsl/glyph.frag"
#include "alluvion/glsl/glyph.vert"
  display->add_shading_program(new ShadingProgram(
      kGlyphVertexShaderStr, kGlyphFragmentShaderStr,
      {
          "projection",
          "text_color",
      },
      [&](ShadingProgram& program, Display& display) {
        glm::mat4 projection =
            glm::ortho(0.0f, static_cast<GLfloat>(display.width_), 0.0f,
                       static_cast<GLfloat>(display.height_));
        glUniformMatrix4fv(program.get_uniform_location("projection"), 1,
                           GL_FALSE, glm::value_ptr(projection));

        std::stringstream time_text;
        time_text << "t = " << std::fixed << std::setprecision(2)
                  << std::setw(6) << t
                  << "s  central speed = " << std::scientific
                  << std::setprecision(4) << std::setw(6) << vx[15] << "m/s";
        std::string text = time_text.str();

        float text_x = display.width_ * 0.4f;
        float text_y = display.height_ * 0.05f;
        float scale = 1.0f;
        glUniform3f(program.get_uniform_location("text_color"), 1.0f, 1.0f,
                    1.0f);
        glBindVertexArray(glyph_vao);

        // iterate through all characters
        std::string::const_iterator c;
        for (c = text.begin(); c != text.end(); c++) {
          Character ch = Characters[*c];

          float xpos = text_x + ch.Bearing.x * scale;
          float ypos = text_y - (ch.Size.y - ch.Bearing.y) * scale;

          float w = ch.Size.x * scale;
          float h = ch.Size.y * scale;
          // update VBO for each character
          float vertices[6][4] = {
              {xpos, ypos + h, 0.0f, 0.0f},    {xpos, ypos, 0.0f, 1.0f},
              {xpos + w, ypos, 1.0f, 1.0f},

              {xpos, ypos + h, 0.0f, 0.0f},    {xpos + w, ypos, 1.0f, 1.0f},
              {xpos + w, ypos + h, 1.0f, 0.0f}};
          // render glyph texture over quad
          glBindTexture(GL_TEXTURE_2D, ch.TextureID);
          // update content of VBO memory
          glBindBuffer(GL_ARRAY_BUFFER, glyph_quad);
          glBufferSubData(
              GL_ARRAY_BUFFER, 0, sizeof(vertices),
              vertices);  // be sure to use glBufferSubData and not glBufferData
          glBindBuffer(GL_ARRAY_BUFFER, 0);

          // render quad
          glDrawArrays(GL_TRIANGLES, 0, 6);
          // now advance cursors for next glyph (note that advance is number of
          // 1/64 pixels)
          text_x +=
              (ch.Advance >> 6) *
              scale;  // bitshift by 6 to get value in pixels (2^6 = 64 (divide
                      // amount of 1/64th pixels by 64 to get amount of pixels))
        }
        glBindTexture(GL_TEXTURE_2D, 0);
      }));
  display->run();
  return 0;
}
