#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <memory>

#include "alluvion/colormaps.hpp"
#include "alluvion/constants.hpp"
#include "alluvion/dg/box_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/display_proxy.hpp"
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
  DisplayProxy<F> display_proxy(display);
  Runner<F> runner;

  F particle_radius = 0.025;
  F kernel_radius = particle_radius * 4;
  F density0 = 1000.0;
  F particle_mass = 0.1;
  F dt = 2e-3;
  F3 gravity = {0._F, -9.81_F, 0._F};
  store.get_cn<F>().set_cubic_discretization_constants();
  store.get_cn<F>().set_kernel_radius(kernel_radius);
  store.get_cn<F>().set_particle_attr(particle_radius, particle_mass, density0);
  store.get_cn<F>().gravity = gravity;
  store.get_cn<F>().viscosity = 0.001;
  store.get_cn<F>().boundary_viscosity = 0.0;

  // rigids
  U max_num_contacts = 512;
  Pile<F> pile(store, max_num_contacts);
  Mesh cube_mesh;
  cube_mesh.set_box(float3{4, 3, 1.5}, 4);
  Mesh sphere_mesh;
  F sphere_radius = 0.1_F;
  sphere_mesh.set_uv_sphere(sphere_radius, 24, 24);
  pile.add(new BoxDistance<F3, F>(F3{4, 3, 1.5}), U3{80, 60, 30}, -1._F, 0,
           cube_mesh, 0._F, 1, 0, F3{1, 1, 1}, F3{0, 1.5, 0}, Q{0, 0, 0, 1},
           Mesh());
  // pile.add(new SphereDistance<F3, F>(sphere_radius), U3{50, 50, 50}, 1._F, 0,
  //          sphere_mesh, 3.2_F, 0.4, 0, F3{1, 1, 1}, F3{0, 0.4, -0},
  //          Q{0, 0, 0, 1}, sphere_mesh);
  pile.build_grids(4 * kernel_radius);
  // pile.build_grids(0.1_F);
  pile.reallocate_kinematics_on_device();
  pile.set_gravity(gravity);
  store.get_cn<F>().contact_tolerance = particle_radius;

  // particles
  U num_particles = 6859;

  // grid
  U3 grid_res{128, 128, 128};
  I3 grid_offset{-64, -64, -64};
  U max_num_particles_per_cell = 64;
  U max_num_neighbors_per_particle = 64;
  store.get_cni().grid_res = grid_res;
  store.get_cni().grid_offset = grid_offset;
  store.get_cni().max_num_particles_per_cell = max_num_particles_per_cell;
  store.get_cni().max_num_neighbors_per_particle =
      max_num_neighbors_per_particle;

  SolverIi<F> solver(runner, pile, store, num_particles, grid_res,
                     max_num_particles_per_cell, max_num_neighbors_per_particle,
                     true);
  solver.num_particles = num_particles;
  solver.dt = dt;
  solver.max_dt = 0.005;
  solver.min_dt = 0.0001;
  solver.cfl = 0.4;
  solver.particle_radius = particle_radius;

  store.copy_cn<F>();
  store.map_graphical_pointers();
  Runner<F>::launch(num_particles, 256, [&](U grid_size, U block_size) {
    create_fluid_block<F3, F>
        <<<grid_size, block_size>>>(*solver.particle_x, num_particles, 0, 0,
                                    F3{-1.95, -0.0, -0.5}, F3{-0.95, 1.0, 0.5});
  });

  store.unmap_graphical_pointers();

  display->camera_.setEye(0.f, 06.00f, 6.0f);
  display->camera_.setCenter(0.f, -0.20f, 0.f);
  display->camera_.update();
  display->update_trackball_camera();

  U frame_id = 0;
  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, {}, [&](ShadingProgram& program, Display& display) {
        // std::cout << "============= frame_id = " << frame_id << std::endl;

        store.map_graphical_pointers();
        // start of simulation loop
        for (U frame_interstep = 0; frame_interstep < 10; ++frame_interstep) {
          solver.step<0>();
        }
        solver.colorize_speed(0, 2);
        store.unmap_graphical_pointers();
        frame_id += 1;
      }));

  GLuint colormap_tex = display_proxy.create_colormap_viridis();
  display_proxy.add_particle_shading_program(
      *solver.particle_x, *solver.particle_normalized_attr, colormap_tex,
      solver.particle_radius, solver);

  display->run();
}
