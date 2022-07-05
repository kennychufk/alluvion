#include <chrono>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <memory>
#include <thread>

#include "alluvion/colormaps.hpp"
#include "alluvion/constants.hpp"
#include "alluvion/dg/box_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/display_proxy.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/solver_i.hpp"
#include "alluvion/solver_ii.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;
using namespace alluvion::dg;

int main(void) {
  Store store;
  Display* display = store.create_display(800, 600, "particle view");
  DisplayProxy<F> display_proxy(display);
  Runner<F> runner;

  F scale_factor = 1;
  F particle_radius = 0.25 * scale_factor;
  F kernel_radius = particle_radius * 4;
  F density0 = 1.0 / (scale_factor * scale_factor * scale_factor);
  F particle_mass = 0.1;
  F dt = 2e-3;
  F3 gravity = {0._F, -98.1_F * scale_factor, 0._F};
  auto cn_cni = store.create_cn<F>();
  Const<F> cn = std::get<0>(cn_cni);
  ConstiN cni = std::get<1>(cn_cni);
  cn.set_kernel_radius(kernel_radius);
  cn.set_particle_attr(particle_radius, particle_mass, density0);
  cn.gravity = gravity;
  cn.viscosity = 0.002 * scale_factor * scale_factor;
  cn.boundary_viscosity = 0.0063 * scale_factor * scale_factor;

  // rigids
  U max_num_contacts = 512;
  const char* pellet_filename =
      "/home/kennychufk/workspace/pythonWs/alluvion-optim/"
      "box-40-30-15-shell.alu";
  U box_num_pellets = std::get<0>(Store::get_alu_info(pellet_filename))[0];
  const char* sphere_pellet_filename =
      "/home/kennychufk/workspace/pythonWs/alluvion-optim/sphere-3.alu";
  U sphere_num_pellets =
      std::get<0>(Store::get_alu_info(sphere_pellet_filename))[0];
  const char* cylinder_pellet_filename =
      "/home/kennychufk/workspace/pythonWs/alluvion-optim/cylinder-3-7.alu";
  U cylinder_num_pellets =
      std::get<0>(Store::get_alu_info(cylinder_pellet_filename))[0];

  Pile<F> pile(store, runner, max_num_contacts, VolumeMethod::pellets,
               box_num_pellets + sphere_num_pellets + cylinder_num_pellets, &cn,
               &cni);

  std::unique_ptr<Variable<1, F3>> pellet_x(
      store.create<1, F3>({box_num_pellets}));
  pellet_x->read_file(pellet_filename);
  pellet_x->scale(scale_factor);
  pile.add(new BoxDistance<F3, F>(scale_factor * F3{40, 30, 15}),
           U3{80, 60, 30}, -1._F, *pellet_x, 0._F, 1, 0, F3{1, 1, 1},
           F3{0, 15 * scale_factor, 0}, Q{0, 0, 0, 1}, Mesh());
  store.remove(*pellet_x);

  F sphere_radius = 3 * scale_factor;
  Mesh sphere_mesh;
  sphere_mesh.set_uv_sphere(sphere_radius, 20, 20);
  std::unique_ptr<Variable<1, F3>> sphere_pellet_x(
      store.create<1, F3>({sphere_num_pellets}));
  sphere_pellet_x->read_file(sphere_pellet_filename);
  sphere_pellet_x->scale(scale_factor);
  pile.add(new SphereDistance<F3, F>(sphere_radius), U3{32, 32, 32}, 1._F,
           *sphere_pellet_x, 50._F * scale_factor * scale_factor * scale_factor,
           0.4, 0, F3{1, 1, 1}, F3{0, 16, -0} * scale_factor, Q{0, 0, 0, 1},
           sphere_mesh);
  store.remove(*sphere_pellet_x);

  F cylinder_radius = 3 * scale_factor;
  F cylinder_height = 7 * scale_factor;
  Mesh cylinder_mesh;
  cylinder_mesh.set_cylinder(cylinder_radius, cylinder_height, 20, 20);
  std::unique_ptr<Variable<1, F3>> cylinder_pellet_x(
      store.create<1, F3>({cylinder_num_pellets}));
  cylinder_pellet_x->read_file(cylinder_pellet_filename);
  cylinder_pellet_x->scale(scale_factor);
  pile.add(new CylinderDistance<F3, F>(cylinder_radius, cylinder_height),
           U3{30, 35, 30}, 1._F, *cylinder_pellet_x,
           30._F * scale_factor * scale_factor * scale_factor, 0.4, 0.5,
           F3{10, 10, 10}, F3{8, 16, -0} * scale_factor, Q{0, 0, 0, 1},
           cylinder_mesh);
  store.remove(*cylinder_pellet_x);

  pile.reallocate_kinematics_on_device();
  pile.set_gravity(gravity);
  cn.contact_tolerance = particle_radius;

  std::cout << "num of pellets = " << sphere_num_pellets << " "
            << box_num_pellets << " = " << pile.num_pellets_ << std::endl;

  // particles
  int block_mode = 0;
  F3 block_min = scale_factor * F3{-19.5, 0.0, -5.0};
  F3 block_max = scale_factor * F3{-9.5, 10.0, 5.0};
  U num_particles = Runner<F>::get_fluid_block_num_particles(
      block_mode, block_min, block_max, particle_radius);

  // grid
  U3 grid_res{128, 128, 128};
  I3 grid_offset{-64, -64, -64};
  cni.grid_res = grid_res;
  cni.grid_offset = grid_offset;
  cni.max_num_particles_per_cell = 64;
  cni.max_num_neighbors_per_particle = 64;

  SolverI<F> solver(runner, pile, store, num_particles, 0, false, false, &cn,
                    &cni, true);
  std::unique_ptr<Variable<1, F>> particle_normalized_attr(
      store.create_graphical<1, F>({num_particles}));
  solver.num_particles = num_particles;
  solver.dt = dt;
  solver.max_dt = 0.005;
  solver.min_dt = 0.0;
  solver.cfl = 0.2;
  solver.min_density_solve = 2;
  solver.max_density_solve = 20;
  solver.particle_radius = particle_radius;

  std::cout << "particle_force shape ";
  for (U shape_item : solver.particle_force->shape_) {
    std::cout << shape_item << " ";
  }
  std::cout << std::endl;

  std::cout << "pellet_id_to_rigid_id_ shape ";
  for (U shape_item : pile.pellet_id_to_rigid_id_->shape_) {
    std::cout << shape_item << " ";
  }
  std::cout << std::endl;

  store.map_graphical_pointers();
  runner.launch_create_fluid_block(*solver.particle_x, num_particles, 0,
                                   particle_radius, block_mode, block_min,
                                   block_max);
  store.unmap_graphical_pointers();

  display->camera_.setEye(0.f, 60.0f * scale_factor, 60.0f * scale_factor);
  display->camera_.setCenter(0.f, -2.0f * scale_factor, 0.f);
  display->camera_.update();
  display->update_trackball_camera();

  U frame_id = 0;
  display->add_shading_program(new ShadingProgram(
      nullptr, nullptr, {}, {}, [&](ShadingProgram& program, Display& display) {
        // std::cout << "============= frame_id = " << frame_id << std::endl;
        // using namespace std::chrono_literals;
        // std::this_thread::sleep_for(2000ms);

        store.map_graphical_pointers();
        // start of simulation loop
        for (U frame_interstep = 0; frame_interstep < 10; ++frame_interstep) {
          solver.step<0>();
        }
        solver.normalize(solver.particle_v.get(),
                         particle_normalized_attr.get(), 0, 2);
        store.unmap_graphical_pointers();
        frame_id += 1;
      }));

  GLuint colormap_tex = display_proxy.create_colormap_viridis();
  display_proxy.add_particle_shading_program(
      *solver.particle_x, *particle_normalized_attr, colormap_tex,
      solver.particle_radius, solver);

  display_proxy.add_pile_shading_program(pile);
  display->run();
  store.remove_graphical(
      dynamic_cast<GraphicalVariable<1, F>&>(*particle_normalized_attr));
  return 0;
}
