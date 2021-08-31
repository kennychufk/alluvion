#include <doctest/doctest.h>

#include <iostream>

#include "alluvion/constants.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;

SCENARIO("testing the runner") {
  GIVEN("a group of particles") {
    Store store;
    U num_particles = 100;
    std::unique_ptr<Variable<1, F3>> var(store.create<1, F3>({num_particles}));
    REQUIRE(var->get_num_primitives() ==
            num_particles * var->get_num_primitives_per_element());
    WHEN("creating fluid block") {
      store.get_cn<F>().set_particle_attr(0.025, 0.0, 0.0);
      store.copy_cn<F>();
      Runner<F>::launch(num_particles, 256, [&](U grid_size, U block_size) {
        create_fluid_block<F3, F><<<grid_size, block_size>>>(
            *var, num_particles, 0, 1, F3{-0.5, 0.0, -0.5}, F3{0.5, 1.0, 0.5});
      });
      THEN("initializes particle x") {
        std::vector<F> copied(num_particles *
                              var->get_num_primitives_per_element());
        var->get_bytes(copied.data(), copied.size() * sizeof(F));
        for (int i = 0; i < num_particles; ++i) {
          std::cout << copied[i * var->get_num_primitives_per_element()] << " "
                    << copied[i * var->get_num_primitives_per_element() + 1]
                    << " "
                    << copied[i * var->get_num_primitives_per_element() + 2]
                    << std::endl;
        }
      }
    }
    store.remove(*var);
  }
}
