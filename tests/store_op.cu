#include <doctest/doctest.h>

#include <iostream>
#include <memory>

#include "alluvion/display.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;

SCENARIO("testing the store") {
  GIVEN("a store with a variable") {
    Store store;
    Variable<1, F3> var = store.create<1, F3>({2});
    REQUIRE(var.get_num_primitives() == 6);
    WHEN("setting from host") {
      std::vector<F> v{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
      var.set_bytes(v.data(), v.size() * sizeof(F));
      THEN("getting from device gives the same data") {
        std::vector<F> copied(6);
        var.get_bytes(copied.data(), copied.size() * sizeof(F));
        CHECK(copied == v);
      }
      THEN("gives the right sum") {
        F result = Runner<F>::sum<F>(var.ptr_, var.get_num_primitives());
        CHECK(result == 21);
      }
    }
  }
  GIVEN("a store with a graphical variable") {
    Store store;
    store.create_display(800, 600, "Test: store_op");
    std::unique_ptr<GraphicalVariable<1, F3>> var(
        store.create_graphical<1, F3>({2}));
    store.map_graphical_pointers();
    REQUIRE(var->get_num_primitives() == 6);
    WHEN("setting from host") {
      std::vector<F> v{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
      var->set_bytes(v.data(), v.size() * sizeof(F));
      THEN("getting from device gives the same data") {
        std::vector<F> copied(6);
        var->get_bytes(copied.data(), copied.size() * sizeof(F));
        CHECK(copied == v);
      }
      THEN("gives the right sum") {
        F result = Runner<F>::sum<F>(var->ptr_, var->get_num_primitives());
        CHECK(result == 21);
      }
    }
    store.unmap_graphical_pointers();
  }
}
