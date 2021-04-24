#include <doctest/doctest.h>

#include <iostream>

#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;

SCENARIO("testing the runner") {
  GIVEN("a group of particles") {
    Store store;
    U num_particles = 100;
    Variable<1, F3> var = store.create<1, F3>({num_particles});
    REQUIRE(var.get_num_elements() == num_particles * 3);
    WHEN("creating fluid block") {
      unsigned int grid_size, block_size;
      Runner::compute_grid_size(num_particles, 256, grid_size, block_size);
      create_fluid_block<F><<<grid_size, block_size>>>(
          var, num_particles, 0, 1, 0.025, -0.5, 0.0, -0.5, 0.5, 1.0, 0.5);
      THEN("initializes particle x") {
        std::vector<F> copied(num_particles * 3);
        var.get_bytes(copied.data(), copied.size() * sizeof(F));
        for (int i = 0; i < num_particles; ++i) {
          std::cout << copied[i * 3] << " " << copied[i * 3 + 1] << " "
                    << copied[i * 3 + 2] << std::endl;
        }
      }
    }
  }
}
