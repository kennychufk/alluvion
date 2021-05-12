#include <vector>

#include "alluvion/allocator.hpp"
#include "alluvion/constants.hpp"
#include "alluvion/dg/gauss_quadrature.hpp"

namespace alluvion {
namespace cnst {
__constant__ F kGridAbscissae[kGridN];
__constant__ F kGridWeights[kGridN];

__constant__ F kernel_radius;
__constant__ F kernel_radius_sqr;
__constant__ F cubic_kernel_k;
__constant__ F cubic_kernel_l;
__constant__ F cubic_kernel_zero;
__constant__ F adhesion_kernel_k;
__constant__ F cohesion_kernel_k;
__constant__ F cohesion_kernel_c;

__constant__ F particle_radius;
__constant__ F particle_vol;
__constant__ F particle_mass;
__constant__ F density0;
__constant__ F viscosity;
__constant__ F vorticity_coeff;
__constant__ F inertia_inverse;
__constant__ F viscosity_omega;
__constant__ F surface_tension_coeff;
__constant__ F surface_tension_boundary_coeff;

__constant__ F3 gravity;

__constant__ I3 neighbor_offsets[kMaxNumCellsToSearch];
__constant__ U num_cells_to_search;
__constant__ U max_num_particles_per_cell;
__constant__ U3 grid_res;
__constant__ I3 grid_offset;
__constant__ F cell_width;
__constant__ U max_num_neighbors_per_particle;

__constant__ U num_boundaries;
__constant__ F contact_tolerance;
__constant__ U max_num_contacts;

__constant__ F wrap_length;
__constant__ F wrap_min;
__constant__ F wrap_max;

void set_cubic_discretization_constants() {
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(kGridAbscissae, &dg::gaussian_abscissae_1[kGridP][0],
                         sizeof(F) * kGridN, 0, cudaMemcpyDefault));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(kGridWeights, &dg::gaussian_weights_1[kGridP][0],
                         sizeof(F) * kGridN, 0, cudaMemcpyDefault));
}

void set_kernel_radius(F h) {
  F h2 = h * h;
  F h3 = h2 * h;
  F tmp_cubic_k = 8.0 / (kPi<F> * h3);
  F tmp_cubic_l = 48.0 / (kPi<F> * h3);
  F tmp_cubic_zero = tmp_cubic_k;
  F tmp_adhesion_kernel_k = 0.007 / pow(h, 3.25);
  F tmp_cohesion_kernel_k = 32.0 / (kPi<F> * h3 * h3 * h3);
  F tmp_cohesion_kernel_c = h3 * h3 / 64.0;

  Allocator::abort_if_error(cudaMemcpyToSymbol(kernel_radius, &h, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(kernel_radius_sqr, &h2, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(cubic_kernel_k, &tmp_cubic_k, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(cubic_kernel_l, &tmp_cubic_l, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(cubic_kernel_zero, &tmp_cubic_zero, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(adhesion_kernel_k, &tmp_adhesion_kernel_k, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(cohesion_kernel_k, &tmp_cohesion_kernel_k, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(cohesion_kernel_c, &tmp_cohesion_kernel_c, sizeof(F)));
}

void set_particle_attr(F radius, F mass, F density) {
  F vol = mass / density;
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(particle_radius, &radius, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(particle_mass, &mass, sizeof(F)));
  Allocator::abort_if_error(cudaMemcpyToSymbol(density0, &density, sizeof(F)));
  Allocator::abort_if_error(cudaMemcpyToSymbol(particle_vol, &vol, sizeof(F)));
}

void set_advanced_fluid_attr(F vis, F vor, F ii, F viso, F st, F stb) {
  Allocator::abort_if_error(cudaMemcpyToSymbol(viscosity, &vis, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(vorticity_coeff, &vor, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(inertia_inverse, &ii, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(viscosity_omega, &viso, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(surface_tension_coeff, &st, sizeof(F)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(surface_tension_boundary_coeff, &stb, sizeof(F)));
}

void init_grid_constants(U3 res, I3 offset) {
  Allocator::abort_if_error(cudaMemcpyToSymbol(grid_res, &res, sizeof(U3)));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(grid_offset, &offset, sizeof(I3)));
}

void set_cell_width(F width) {
  Allocator::abort_if_error(cudaMemcpyToSymbol(cell_width, &width, sizeof(F)));
}

void set_search_range(F search_radius_relative_to_cell_width) {
  std::vector<I3> offsets;
  F relative_radius_squared = search_radius_relative_to_cell_width *
                              search_radius_relative_to_cell_width;
  int evaluation_extent =
      static_cast<int>(ceil(search_radius_relative_to_cell_width));
  for (int i = -evaluation_extent; i <= evaluation_extent; ++i) {
    for (int j = -evaluation_extent; j <= evaluation_extent; ++j) {
      for (int k = -evaluation_extent; k <= evaluation_extent; ++k) {
        if (static_cast<F>(i * i + j * j + k * k) <= relative_radius_squared) {
          offsets.push_back(I3{i, j, k});
        }
      }
    }
  }
  U num_cells = offsets.size();
  if (num_cells > kMaxNumCellsToSearch) {
    std::cerr << "Num of cells to search exceeeds the limit "
              << kMaxNumCellsToSearch << std::endl;
    abort();
  }
  Allocator::abort_if_error(cudaMemcpyToSymbol(neighbor_offsets, offsets.data(),
                                               sizeof(I3) * num_cells));
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(num_cells_to_search, &num_cells, sizeof(U)));
}
void set_max_num_particles_per_cell(U n) {
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(max_num_particles_per_cell, &n, sizeof(U)));
}
void set_max_num_neighbors_per_particle(U n) {
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(max_num_neighbors_per_particle, &n, sizeof(U)));
}
void set_gravity(F3 g) {
  Allocator::abort_if_error(cudaMemcpyToSymbol(gravity, &g, sizeof(F3)));
}
void set_num_boundaries(U n) {
  Allocator::abort_if_error(cudaMemcpyToSymbol(num_boundaries, &n, sizeof(U)));
}
void set_contact_tolerance(F tolerance) {
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(contact_tolerance, &tolerance, sizeof(F)));
}
void set_max_num_contacts(U n) {
  Allocator::abort_if_error(
      cudaMemcpyToSymbol(max_num_contacts, &n, sizeof(U)));
}
void set_wrap_length(F l) {
  F wmin = l * -0.5_F;
  F wmax = l * 0.5_F;
  Allocator::abort_if_error(cudaMemcpyToSymbol(wrap_length, &l, sizeof(F)));
  Allocator::abort_if_error(cudaMemcpyToSymbol(wrap_min, &wmin, sizeof(F)));
  Allocator::abort_if_error(cudaMemcpyToSymbol(wrap_max, &wmax, sizeof(F)));
}
}  // namespace cnst
}  // namespace alluvion
