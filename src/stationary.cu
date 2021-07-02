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

  // rigids
  U max_num_contacts = 512;
  Pile<F3, Q, F> pile(store, max_num_contacts);
  Mesh cube_mesh;
  cube_mesh.set_box(float3{4, 3, 1.5}, 4);
  Mesh sphere_mesh;
  F sphere_radius = 0.1_F;
  sphere_mesh.set_uv_sphere(sphere_radius, 24, 24);
  pile.add(new BoxDistance<F3, F>(F3{4, 3, 1.5}), U3{80, 60, 30}, -1._F, 0,
           cube_mesh, 0._F, 1, 0, 0.0, F3{1, 1, 1}, F3{0, 1.5, 0},
           Q{0, 0, 0, 1}, Mesh());
  // pile.add(new SphereDistance<F3, F>(sphere_radius), U3{50, 50, 50}, 1._F, 0,
  //          sphere_mesh, 3.2_F, 0.4, 0, 0.2, F3{1, 1, 1}, F3{0, 0.4, -0},
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
  Runner<F>::launch(num_particles, 256, [&](U grid_size, U block_size) {
    create_fluid_block<F3, F>
        <<<grid_size, block_size>>>(*particle_x, num_particles, 0, 0,
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
          solver_ii.step<0>();
        }
        solver_ii.colorize_speed(0, 2);
        store.unmap_graphical_pointers();
        frame_id += 1;
      }));

  GLuint colormap_tex = display_proxy.create_colormap_viridis();
  display_proxy.add_particle_shading_program(
      particle_x->vbo_, particle_normalized_attr->vbo_, colormap_tex,
      solver_ii.particle_radius, solver_ii);

  display->run();
}
