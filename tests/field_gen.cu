#include <doctest/doctest.h>

#include <vector>

#include "alluvion/constants.hpp"
#include "alluvion/dg/cubic_lagrange_discrete_grid.hpp"
#include "alluvion/dg/gauss_quadrature.hpp"
#include "alluvion/dg/sph_kernels.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;
using namespace alluvion::dg;

SCENARIO("testing volume field generation") {
  GIVEN("a host implementation") {
    std::array<unsigned int, 3> resolution_array{11, 11, 11};
    F map_thickness = 0.1;
    F particle_radius = 0.25;
    F sign = 1.0;
    F r = 1.25;
    F factor = 1.0;
    F margin = 4 * r + map_thickness;
    Vector3r<F> domain_min(-30 - margin, -20 - margin, -40 - margin);
    Vector3r<F> domain_max(30 + margin, 20 + margin, 40 + margin);
    AlignedBox3r<F> domain(domain_min, domain_max);
    CubicLagrangeDiscreteGrid<F> grid(domain, resolution_array);
    grid.addFunction(
        [sign, map_thickness, particle_radius](Vector3r<F> const &xi) {
          F signed_distance_from_ellipsoid = xi(0) * xi(0) / (25 * 25) +
                                             xi(1) * xi(1) / (15 * 15) +
                                             xi(2) * xi(2) / (35 * 35) - 1.0;
          return sign * (signed_distance_from_ellipsoid - map_thickness -
                         particle_radius * 0.5);
        });
    CubicKernel<F> cubic_kernel;
    cubic_kernel.setRadius(r);
    auto int_domain =
        AlignedBox3r<F>(Vector3r<F>::Constant(-r), Vector3r<F>::Constant(r));
    grid.addFunction(
        [&](Vector3r<F> const &x) {
          auto dist = grid.interpolate(0u, x);
          if (dist > (1.0 + 1.0 /*/ factor*/) * r) {
            return 0.0;
          }

          auto integrand = [&grid, &x, &r, &factor,
                            &cubic_kernel](Vector3r<F> const &xi) -> F {
            if (xi.squaredNorm() > r * r) return 0.0;

            auto dist = grid.interpolate(0u, x + xi);

            if (dist <= 0.0) return 1.0 - 0.1 * dist / r;
            if (dist < 1.0 / factor * r)
              return cubic_kernel.W(factor * dist) / cubic_kernel.W_zero();
            return 0.0;
          };
          return 0.8 * GaussQuadrature<F>::integrate(integrand, int_domain, 30);
        },
        false);
    WHEN("a device implementation is constructed") {
      Runner<F> runner;
      CubicLagrangeDiscreteGrid<F> distance_grid_prerequisite(domain,
                                                              resolution_array);
      distance_grid_prerequisite.addFunction([](Vector3r<F> const &xi) {
        F signed_distance_from_ellipsoid = xi(0) * xi(0) / (25 * 25) +
                                           xi(1) * xi(1) / (15 * 15) +
                                           xi(2) * xi(2) / (35 * 35) - 1.0;
        return signed_distance_from_ellipsoid;
      });
      // copy attributes
      U num_nodes = distance_grid_prerequisite.node_data()[0].size();
      F3 domain_min = {domain.min()(0), domain.min()(1), domain.min()(2)};
      F3 domain_max = {domain.max()(0), domain.max()(1), domain.max()(2)};
      U3 resolution = make_uint3(resolution_array[0], resolution_array[1],
                                 resolution_array[2]);
      F3 cell_size = {distance_grid_prerequisite.cellSize()(0),
                      distance_grid_prerequisite.cellSize()(1),
                      distance_grid_prerequisite.cellSize()(2)};
      // allocate device memory
      Store store;
      std::unique_ptr<Variable<1, F>> distance_nodes(
          store.create<1, F>({num_nodes}));
      std::unique_ptr<Variable<1, F>> volume_nodes(
          store.create<1, F>({num_nodes}));
      distance_nodes->set_bytes(
          distance_grid_prerequisite.node_data()[0].data(),
          num_nodes * sizeof(F));

      // set constants
      store.get_cn<F>().set_particle_attr(particle_radius, 0.2, 1.0);
      store.get_cn<F>().set_kernel_radius(r);
      store.get_cn<F>().set_cubic_discretization_constants();
      store.copy_cn<F>();

      runner.launch(
          num_nodes,
          [&](U grid_size, U block_size) {
            update_volume_field<<<grid_size, block_size>>>(
                *volume_nodes, *distance_nodes, domain_min, domain_max,
                resolution, cell_size, num_nodes, 0, sign, map_thickness);
          },
          "update_volume_field", update_volume_field<F3, F>);

      std::vector<F> device_volume_nodes_copied(num_nodes);
      volume_nodes->get_bytes(device_volume_nodes_copied.data(),
                              device_volume_nodes_copied.size() * sizeof(F));
      for (U l = 0; l < num_nodes; ++l) {
        if (device_volume_nodes_copied[l] > 1e-5) {
          std::cout << l << " " << device_volume_nodes_copied[l] << " "
                    << grid.node_data()[1][l] << std::endl;
        }
        CHECK(device_volume_nodes_copied[l] ==
              doctest::Approx(grid.node_data()[1][l]));
      }
    }
  }
}
