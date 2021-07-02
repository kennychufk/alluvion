#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <memory>

#include "alluvion/constants.hpp"
#include "alluvion/dg/box_distance.hpp"
#include "alluvion/dg/cylinder_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/display_proxy.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/solver_df.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;
using namespace alluvion::dg;

int main(void) {
  Store store;
  Display* display = store.create_display(800, 600, "particle view");
  DisplayProxy<F> display_proxy(display);
  Runner<F> runner;

  F particle_radius = 0.00125;
  F kernel_radius = particle_radius * 4;
  F density0 = 1000.0;
  F cubical_particle_volume =
      8 * particle_radius * particle_radius * particle_radius;
  F volume_relative_to_cube = 0.8_F;
  F particle_mass =
      cubical_particle_volume * volume_relative_to_cube * density0;
  F dt = 1e-4;
  F3 gravity = {0._F, -9.81_F, 0._F};
  store.get_cn<F>().set_cubic_discretization_constants();
  store.get_cn<F>().set_kernel_radius(kernel_radius);
  store.get_cn<F>().set_particle_attr(particle_radius, particle_mass, density0);
  store.get_cn<F>().gravity = gravity;
  store.get_cn<F>().viscosity = 3.54008928e-06;

  // rigids
  F boundary_viscosity = 6.71368218e-06;
  U max_num_contacts = 512;
  Pile<F3, Q, F> pile(store, max_num_contacts);
  Mesh cube_mesh;
  cube_mesh.set_box(float3{0.24, 0.24, 0.24}, 4);
  Mesh sphere_mesh;
  F sphere_radius = 0.20_F;
  sphere_mesh.set_uv_sphere(sphere_radius, 24, 24);
  pile.add(new BoxDistance<F3, F>(F3{0.24, 0.24, 0.24}), U3{64, 64, 64}, -1._F,
           0, cube_mesh, 0._F, 1, 0, boundary_viscosity, F3{1, 1, 1},
           F3{0, 0.12, 0}, Q{0, 0, 0, 1}, cube_mesh);
  // pile.add(new SphereDistance<F3, F>(sphere_radius), U3{100, 100, 100}, 1._F,
  // 0,
  //          sphere_mesh, 20.0_F, 0.4, 0, 0, F3{1, 1, 1}, F3{0, 0.4, -0},
  //          Q{0, 0, 0, 1}, sphere_mesh);
  F cylinder_radius = 3.00885e-3_F - particle_radius;
  F cylinder_height = 38.5e-3_F - particle_radius * 2;
  F cylinder_comy = -8.8521e-3_F;
  Mesh cylinder_mesh;
  cylinder_mesh.set_cylinder(cylinder_radius, cylinder_height, 24, 24);
  cylinder_mesh.translate(float3{0, -cylinder_comy, 0});
  pile.add(new CylinderDistance<F3, F>(cylinder_radius, cylinder_height,
                                       cylinder_comy),
           U3{32, 256, 32}, 1, 0, cylinder_mesh, 1.07e-3_F,  // mass
           0,                                                // restitution
           0,                                                // friction
           boundary_viscosity,                            // boundary_viscosity
           F3{7.91134e-8_F, 2.94462e-9_F, 7.91134e-8_F},  // inertia ,
           F3{0, 0.15, 0.06}, Q{0, 0, 0, 1}, cylinder_mesh);
  pile.build_grids(4 * kernel_radius);
  // pile.build_grids(0.1_F);
  pile.reallocate_kinematics_on_device();
  pile.set_gravity(gravity);
  store.get_cn<F>().contact_tolerance = particle_radius;
  store.get_cn<F>().dfsph_factor_epsilon = 1e-5;

  // particles
  U num_particles = 380000;
  // U num_particles = 10;

  // grid
  U3 grid_res{128, 128, 128};
  I3 grid_offset{-64, 0, -64};
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
  Variable<1, F> particle_dfsph_factor = store.create<1, F>({num_particles});
  Variable<1, F> particle_kappa = store.create<1, F>({num_particles});
  Variable<1, F> particle_kappa_v = store.create<1, F>({num_particles});
  Variable<1, F> particle_density_adv = store.create<1, F>({num_particles});
  Variable<4, Q> pid = store.create<4, Q>(
      {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell});
  Variable<3, U> pid_length =
      store.create<3, U>({grid_res.x, grid_res.y, grid_res.z});
  Variable<2, Q> particle_neighbors =
      store.create<2, Q>({num_particles, max_num_neighbors_per_particle});
  Variable<1, U> particle_num_neighbors = store.create<1, U>({num_particles});

  SolverDf<F3, Q, F> solver_df(
      runner, pile, *particle_x, *particle_normalized_attr, particle_v,
      particle_a, particle_density, particle_boundary_xj,
      particle_boundary_volume, particle_force, particle_torque,
      particle_cfl_v2, particle_dfsph_factor, particle_kappa, particle_kappa_v,
      particle_density_adv, pid, pid_length, particle_neighbors,
      particle_num_neighbors);
  solver_df.num_particles = num_particles;
  solver_df.dt = dt;
  solver_df.max_dt = 0.005;
  solver_df.min_dt = 0.0;
  solver_df.cfl = 0.1;
  solver_df.particle_radius = particle_radius;

  store.copy_cn<F>();
  store.map_graphical_pointers();
  Runner<F>::launch(num_particles, 256, [&](U grid_size, U block_size) {
    create_fluid_block<F3, F><<<grid_size, block_size>>>(
        *particle_x, num_particles, 0, 0,
        F3{-0.12 + particle_radius * 2, particle_radius * 2,
           -0.12 + particle_radius * 2},
        F3{0.12 - particle_radius * 2, 0.12 - particle_radius * 2,
           0.12 - particle_radius * 2});
  });

  store.unmap_graphical_pointers();

  display->camera_.setEye(0.f, 0.06f, 0.40f);
  display->camera_.setCenter(0.f, 0.06f, 0.f);
  display->camera_.update();
  display->update_trackball_camera();

  U frame_id = 0;
  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, {}, [&](ShadingProgram& program, Display& display) {
        // std::cout << "============= frame_id = " << frame_id << std::endl;

        store.map_graphical_pointers();
        // start of simulation loop
        for (U frame_interstep = 0; frame_interstep < 10; ++frame_interstep) {
          solver_df.step<0, 0>();
        }
        solver_df.colorize_kappa_v(-0.002_F, 0.0_F);
        // solver_df.colorize_speed(0, 2);
        store.unmap_graphical_pointers();
        frame_id += 1;
      }));

  GLuint colormap_tex = display_proxy.create_colormap_viridis();
  display_proxy.add_particle_shading_program(
      particle_x->vbo_, particle_normalized_attr->vbo_, colormap_tex,
      solver_df.particle_radius, solver_df);

  display_proxy.add_pile_shading_program(pile);
  display->run();
}
