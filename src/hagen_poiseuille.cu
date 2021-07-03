#include <cassert>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <limits>
#include <random>

#include "alluvion/colormaps.hpp"
#include "alluvion/constants.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/display_proxy.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/solver_df.hpp"
#include "alluvion/store.hpp"
#include "alluvion/typesetter.hpp"

using namespace alluvion;
using namespace alluvion::dg;

int main(void) {
  Store store;
  Display* display = store.create_display(800, 600, "particle view");
  DisplayProxy<F> display_proxy(display);
  Runner<F> runner;

  F particle_radius = 0.0025_F;
  F kernel_radius = particle_radius * 4.0_F;
  F density0 = 1000.0_F;
  F cubical_particle_volume =
      8 * particle_radius * particle_radius * particle_radius;
  F volume_relative_to_cube = 0.8_F;
  F particle_mass =
      cubical_particle_volume * volume_relative_to_cube * density0;

  store.get_cn<F>().set_cubic_discretization_constants();
  store.get_cn<F>().set_kernel_radius(kernel_radius);
  store.get_cn<F>().set_particle_attr(particle_radius, particle_mass, density0);
  store.get_cn<F>().axial_gravity = -10.0_F;
  store.get_cn<F>().radial_gravity = 5.0_F;
  store.get_cn<F>().boundary_epsilon = 1e-9_F;
  F viscosity = 5e-6_F;
  F vorticity = 0.01_F;
  F inertia_inverse = 0.1_F;
  F viscosity_omega = 0.5_F;
  F surface_tension_coeff = 0.05_F;
  F surface_tension_boundary_coeff = 0.01_F;
  store.get_cn<F>().viscosity = viscosity;
  store.get_cn<F>().boundary_viscosity = viscosity * 1.5_F;

  I kM = 5;
  F cylinder_length = 2._F * kM * kernel_radius;
  I kQ = 5;
  F R = kernel_radius * kQ;

  const char* font_filename = "/usr/share/fonts/truetype/freefont/FreeMono.ttf";
  Typesetter typesetter(display, font_filename, 0, 30);
  typesetter.load_ascii();

  display->camera_.setEye(0._F, 0._F, R * 6._F);
  display->camera_.setClipPlanes(particle_radius * 10._F, R * 20._F);
  display->update_trackball_camera();

  GLuint glyph_quad = display->create_dynamic_array_buffer<float4>(6, nullptr);

  // rigids
  F restitution = 1._F;
  F friction = 0._F;
  U max_num_contacts = 512;
  Pile<F> pile(store, max_num_contacts);
  pile.add(new InfiniteCylinderDistance<F3, F>(R), U3{64, 1, 64}, -1._F, 0,
           Mesh(), 0._F, restitution, friction, F3{1, 1, 1}, F3{0, 0, 0},
           Q{0, 0, 0, 1}, Mesh());
  pile.build_grids(4 * kernel_radius);
  pile.reallocate_kinematics_on_device();
  store.get_cn<F>().contact_tolerance = particle_radius;

  // particles
  U naive_num_particles =
      static_cast<U>(kPi<F> * (R - particle_radius) * (R - particle_radius) *
                     cylinder_length * density0 / particle_mass);
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

  SolverDf<F> solver(runner, pile, store, max_num_particles, grid_res,
                     max_num_particles_per_cell, max_num_neighbors_per_particle,
                     true);
  solver.dt = 1e-3_F;
  solver.max_dt = 1e-3_F;
  solver.min_dt = 0.0_F;
  solver.cfl = 2e-2_F;
  solver.particle_radius = particle_radius;

  U initial_num_particles = naive_num_particles;
  F slice_distance = particle_radius * 2._F;
  U num_slices = static_cast<U>(cylinder_length / slice_distance);
  U num_particles_per_slice = initial_num_particles / num_slices;
  initial_num_particles = num_particles_per_slice * num_slices;

  store.copy_cn<F>();
  store.map_graphical_pointers();
  Runner<F>::launch(initial_num_particles, 256, [&](U grid_size, U block_size) {
    create_fluid_cylinder<F3, F><<<grid_size, block_size>>>(
        *solver.particle_x, initial_num_particles, R - particle_radius * 2._F,
        num_particles_per_slice, particle_radius * 2._F,
        cylinder_length * -0.5_F);
  });
  store.unmap_graphical_pointers();
  solver.num_particles = initial_num_particles;

  U step_id = 0;
  F t = 0;
  F max_density_error = std::numeric_limits<F>::max();
  F min_density_error = std::numeric_limits<F>::max();
  F naive_filled_percentage = 0;
  bool should_close = false;

  F max_particle_speed = 99.9_F;
  F min_particle_speed = 99.9_F;
  F sum_particle_velocity_components = 99.9_F;
  F last_stationary_t = 0._F;

  F next_emission_t = 5.0_F;
  F last_emission_t = 0._F;
  bool speed_ready_before_emission = false;

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<F> d{0, 0.05};

  U2 target_num_particles_range = U2{0, 7213};
  U last_saved_num_particles = 0;

  // density sample points
  U num_sample_slices = 32;
  U num_samples_per_slice = 16;
  U num_samples = num_sample_slices * num_samples_per_slice;
  std::unique_ptr<Variable<1, F3>> sample_x(store.create<1, F3>({num_samples}));
  std::unique_ptr<Variable<1, F>> sample_data1(
      store.create<1, F>({num_samples}));
  std::vector<F> sample_data1_host(num_samples);
  std::unique_ptr<Variable<2, Q>> sample_neighbors(
      store.create<2, Q>({num_samples, max_num_neighbors_per_particle}));
  std::unique_ptr<Variable<1, U>> sample_num_neighbors(
      store.create<1, U>({num_samples}));
  Runner<F>::launch(initial_num_particles, 256, [&](U grid_size, U block_size) {
    create_fluid_cylinder<F3, F><<<grid_size, block_size>>>(
        *sample_x, num_samples, R - particle_radius * 2._F,
        num_samples_per_slice, cylinder_length / num_sample_slices,
        cylinder_length * -0.5_F);
  });
  std::vector<F3> sample_x_host(num_samples);
  sample_x->get_bytes(sample_x_host.data());
  assert(sample_x_host[0].x == sample_x_host[2].z == 0);
  assert(sample_x_host[0].y == -cylinder_length / 2);

  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, {}, [&](ShadingProgram& program, Display& display) {
        if (should_close) {
          return;
        }

        store.map_graphical_pointers();
        for (U frame_interstep = 0; frame_interstep < 10; ++frame_interstep) {
          if (t > 10._F && solver.num_particles == 2704 &&
              t - last_stationary_t > 0.2_F) {
            // particle_x->write_file("x5-2704.alu", solver.num_particles);
            // should_close = true;
          }
          if (min_particle_speed > 2._F || (step_id % 10000 == 0)) {
            solver.particle_v->set_zero();
            solver.particle_dfsph_factor->set_zero();
            solver.particle_kappa->set_zero();
            solver.particle_kappa_v->set_zero();
            solver.particle_density_adv->set_zero();
            last_stationary_t = t;
            std::cout << "last stationary t = " << last_stationary_t
                      << std::endl;
          }
          if (!speed_ready_before_emission) {
            if (min_density_error < -0.5) {
              if (max_particle_speed < 1e-1_F)
                speed_ready_before_emission = true;
            } else if (min_density_error < -0.05) {
              if (max_particle_speed < 5e-3_F)
                speed_ready_before_emission = true;
            } else {
              if (max_particle_speed < 1e-2_F && min_density_error > -1e-2 &&
                  max_density_error < 1e-2) {
                speed_ready_before_emission = true;
              }
            }
          }
          if (solver.num_particles >= target_num_particles_range.x &&
              solver.num_particles <= target_num_particles_range.y &&
              solver.num_particles != last_saved_num_particles) {
            speed_ready_before_emission = false;
            if (max_particle_speed < 2e-3 && (t - last_stationary_t > 4)) {
              std::stringstream filename_stream;
              filename_stream << "x" << kQ << "-" << kM << "-"
                              << solver.num_particles << ".alu";
              std::string filename = filename_stream.str();
              solver.particle_x->write_file(filename.c_str(),
                                            solver.num_particles);
              last_saved_num_particles = solver.num_particles;
            }
          }
          if (t > next_emission_t && speed_ready_before_emission &&
              solver.num_particles < max_num_particles) {
            F3 new_particle_pos{0, cylinder_length / 2 - 1e-4, 0};
            F3 new_particle_v{d(gen), d(gen), d(gen)};
            next_emission_t = t + particle_radius * 2 / length(new_particle_v);
            solver.particle_x->set_bytes(&new_particle_pos, sizeof(F3),
                                         sizeof(F3) * solver.num_particles);
            solver.particle_v->set_bytes(&new_particle_v, sizeof(F3),
                                         sizeof(F3) * solver.num_particles);
            ++solver.num_particles;
            last_emission_t = t;
            speed_ready_before_emission = false;
          }

          solver.step<1, 1>();

          t += solver.dt;
          step_id += 1;

          max_density_error =
              Runner<F>::max(*solver.particle_density, solver.num_particles) /
                  density0 -
              1;
          min_density_error =
              Runner<F>::min(*solver.particle_density, solver.num_particles) /
                  density0 -
              1;
          max_particle_speed = sqrt(
              Runner<F>::max(*solver.particle_cfl_v2, solver.num_particles));
          min_particle_speed = sqrt(
              Runner<F>::min(*solver.particle_cfl_v2, solver.num_particles));
          sum_particle_velocity_components = Runner<F>::sum<F>(
              solver.particle_v->ptr_, solver.particle_v->get_num_primitives());

          F expected_total_volume = kPi<F> * (R - particle_radius) *
                                    (R - particle_radius) * cylinder_length;
          naive_filled_percentage = solver.num_particles * particle_mass /
                                    density0 / expected_total_volume;
          if (step_id % 100 == 0) {
            solver.pid_length->set_zero();
            Runner<F>::launch(solver.num_particles, 256,
                              [&](U grid_size, U block_size) {
                                update_particle_grid<<<grid_size, block_size>>>(
                                    *solver.particle_x, *solver.pid,
                                    *solver.pid_length, solver.num_particles);
                              });
            Runner<F>::launch(num_samples, 256, [&](U grid_size, U block_size) {
              make_neighbor_list<1><<<grid_size, block_size>>>(
                  *sample_x, *solver.pid, *solver.pid_length, *sample_neighbors,
                  *sample_num_neighbors, num_samples);
            });
            Runner<F>::launch(
                solver.num_particles, 256, [&](U grid_size, U block_size) {
                  compute_density<<<grid_size, block_size>>>(
                      *solver.particle_x, *solver.particle_neighbors,
                      *solver.particle_num_neighbors, *solver.particle_density,
                      *solver.particle_boundary_xj,
                      *solver.particle_boundary_volume, solver.num_particles);
                });
            Runner<F>::launch(num_samples, 256, [&](U grid_size, U block_size) {
              sample_fluid<<<grid_size, block_size>>>(
                  *sample_x, *solver.particle_x, *solver.particle_density,
                  *solver.particle_density, *sample_neighbors,
                  *sample_num_neighbors, *sample_data1, num_samples);
            });
            sample_data1->get_bytes(sample_data1_host.data());
            for (U i = 0; i < num_samples; i += 127) {
              std::cout << sample_data1_host[i] << " ";
            }
            std::cout << std::endl;
          }
        }
        solver.colorize_speed(0, 2.0);
        store.unmap_graphical_pointers();
      }));

  GLuint colormap_tex = display_proxy.create_colormap_viridis();
  display_proxy.add_particle_shading_program(
      *solver.particle_x, *solver.particle_normalized_attr, colormap_tex,
      solver.particle_radius, solver);

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
        time_text << "num_particles = " << solver.num_particles << "("
                  << std::fixed << std::setprecision(3) << std::setw(5)
                  << naive_filled_percentage << " t: " << std::fixed
                  << std::setprecision(3) << std::setw(6) << t
                  << " dt: " << std::scientific << std::setprecision(3)
                  << std::setw(6) << solver.dt << " d: (" << std::scientific
                  << std::setprecision(3) << std::setw(6) << min_density_error
                  << "," << std::scientific << std::setprecision(3)
                  << std::setw(6) << max_density_error << ") v=("
                  << std::setw(6) << min_particle_speed << ","
                  << max_particle_speed << ")";
        std::string text = time_text.str();

        typesetter.start(display.width_ * 0.02f, display.height_ * 0.05f, 1.0f);
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
  display->run();
}
