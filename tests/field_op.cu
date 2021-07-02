#include <doctest/doctest.h>

#include "alluvion/dg/cubic_lagrange_discrete_grid.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;
using namespace alluvion::dg;

__global__ void get_node_positions(Variable<1, F3> positions, F3 domain_min,
                                   U3 resolution, F3 cell_size, U num_nodes) {
  forThreadMappedToElement(num_nodes, [&](U l) {
    positions(l) = index_to_node_position(domain_min, resolution, cell_size, l);
  });
}
SCENARIO("testing index-to-node") {
  GIVEN("A valid domain and resolution") {
    AlignedBox3r<F> domain(Vector3r<F>(-23.1, 12.23, -31.44),
                           Vector3r<F>(45.1, 103.2, 12.4));
    std::array<unsigned int, 3> resolution_array{11, 11, 11};
    WHEN("composing a grid") {
      CubicLagrangeDiscreteGrid<F> grid(domain, resolution_array);
      grid.addFunction([](Vector3r<F> const& xi) { return xi(0); });
      U num_nodes = grid.node_data()[0].size();

      F3 domain_min{domain.min()(0), domain.min()(1), domain.min()(2)};
      U3 resolution = make_uint3(resolution_array[0], resolution_array[1],
                                 resolution_array[2]);
      F3 cell_size{grid.cellSize()(0), grid.cellSize()(1), grid.cellSize()(2)};
      THEN(
          "gives the same node positions for device and host implementations") {
        Store store;
        Variable<1, F3> device_positions = store.create<1, F3>({num_nodes});

        Runner<F>::launch(num_nodes, 256, [&](U grid_size, U block_size) {
          get_node_positions<<<grid_size, block_size>>>(
              device_positions, domain_min, resolution, cell_size, num_nodes);
        });

        std::vector<F3> device_positions_copied(num_nodes);
        device_positions.get_bytes(device_positions_copied.data(),
                                   device_positions_copied.size() * sizeof(F3));

        for (U l = 0; l < num_nodes; ++l) {
          Vector3r<F> host_position = grid.indexToNodePosition(l);
          F3 device_position = device_positions_copied[l];
          CHECK(device_position.x == doctest::Approx(host_position(0)));
          CHECK(device_position.y == doctest::Approx(host_position(1)));
          CHECK(device_position.z == doctest::Approx(host_position(2)));
        }
      }
    }
  }
}
