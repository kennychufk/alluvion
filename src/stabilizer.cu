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

int main(int argc, char* argv[]) {
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
  store.get_cn<F>().gravity = F3{0, 0.0, 0};
  store.get_cn<F>().boundary_epsilon = 1e-9_F;
  F viscosity = 5e-6_F;
  F vorticity = 0.01_F;
  F inertia_inverse = 0.1_F;
  F viscosity_omega = 0.5_F;
  F surface_tension_coeff = 0.05_F;
  F surface_tension_boundary_coeff = 0.01_F;
  store.get_cn<F>().viscosity = viscosity;
  store.get_cn<F>().boundary_viscosity = viscosity * 1.5_F;

  I kM = 2;
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
  InfiniteCylinderDistance<F3, F> cylinder_distance(R);
  pile.add(&cylinder_distance, U3{64, 1, 64}, -1._F, 0, Mesh(), 0._F,
           restitution, friction, F3{1, 1, 1}, F3{0, 0, 0}, Q{0, 0, 0, 1},
           Mesh());
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
  store.get_cni().grid_res = grid_res;
  store.get_cni().grid_offset = grid_offset;
  store.get_cni().max_num_particles_per_cell = 64;
  store.get_cni().max_num_neighbors_per_particle = 64;
  store.get_cn<F>().set_wrap_length(grid_res.y * kernel_radius);

  SolverDf<F> solver(runner, pile, store, max_num_particles, grid_res, false,
                     false, true);
  std::unique_ptr<Variable<1, F>> particle_normalized_attr(
      store.create_graphical<1, F>({max_num_particles}));
  solver.dt = 1e-3_F;
  solver.max_dt = 1e-4_F;
  solver.min_dt = 0.0_F;
  solver.cfl = 2e-3_F;
  solver.particle_radius = particle_radius;

  store.copy_cn<F>();
  store.map_graphical_pointers();
  solver.num_particles = solver.particle_x->read_file(argv[1]);
  store.unmap_graphical_pointers();

  U step_id = 0;
  F t = 0;
  F max_density_error = std::numeric_limits<F>::max();
  F min_density_error = std::numeric_limits<F>::max();
  bool should_close = false;

  F max_particle_speed = 99.9_F;
  F min_particle_speed = 99.9_F;
  F sum_particle_velocity_components = 99.9_F;
  F last_stationary_t = 0._F;

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<F> d{0, 0.05};

  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, {}, [&](ShadingProgram& program, Display& display) {
        if (should_close) {
          return;
        }

        store.map_graphical_pointers();
        for (U frame_interstep = 0; frame_interstep < 10; ++frame_interstep) {
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
          if (t > 6) {
            solver.particle_x->write_file(argv[2], solver.num_particles);
            should_close = true;
          }

          solver.step<1, 0>();

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
        }
        solver.normalize(solver.particle_v.get(),
                         particle_normalized_attr.get(), 0, 2);
        store.unmap_graphical_pointers();
      }));

  GLuint colormap_tex = display_proxy.create_colormap_viridis();
  display_proxy.add_particle_shading_program(
      *solver.particle_x, *particle_normalized_attr, colormap_tex,
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
        time_text << "num_particles = " << solver.num_particles
                  << " t: " << std::fixed << std::setprecision(3)
                  << std::setw(6) << t << " dt: " << std::scientific
                  << std::setprecision(3) << std::setw(6) << solver.dt
                  << " d: (" << std::scientific << std::setprecision(3)
                  << std::setw(6) << min_density_error << "," << std::scientific
                  << std::setprecision(3) << std::setw(6) << max_density_error
                  << ") v=(" << std::setw(6) << min_particle_speed << ","
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
  // }}}
}
