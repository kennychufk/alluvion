#ifndef ALLUVION_CONSTANTS_HPP
#define ALLUVION_CONSTANTS_HPP

#include <limits>

#include "alluvion/data_type.hpp"

namespace alluvion {
template <class T>
constexpr T kPi = T(
    3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651e+00L);
constexpr F kFMax = std::numeric_limits<F>::max();

constexpr U kGridN = 16;
constexpr U kGridP = 30;
constexpr U kMaxNumCellsToSearch = 128;
// a separate namespace for constant memory and its setters
namespace cnst {
extern __constant__ F kGridAbscissae[kGridN];
extern __constant__ F kGridWeights[kGridN];

extern __constant__ F kernel_radius;
extern __constant__ F kernel_radius_sqr;
extern __constant__ F cubic_kernel_k;
extern __constant__ F cubic_kernel_l;
extern __constant__ F cubic_kernel_zero;
extern __constant__ F adhesion_kernel_k;
extern __constant__ F cohesion_kernel_k;
extern __constant__ F cohesion_kernel_c;

extern __constant__ F particle_radius;
extern __constant__ F particle_vol;
extern __constant__ F particle_mass;
extern __constant__ F density0;
extern __constant__ F viscosity;
extern __constant__ F vorticity_coeff;
extern __constant__ F inertia_inverse;
extern __constant__ F viscosity_omega;
extern __constant__ F surface_tension_coeff;
extern __constant__ F surface_tension_boundary_coeff;

extern __constant__ F gravity;

extern __constant__ I3 neighbor_offsets[kMaxNumCellsToSearch];
extern __constant__ U num_cells_to_search;
extern __constant__ U max_num_particles_per_cell;
extern __constant__ U3 grid_res;
extern __constant__ I3 grid_offset;
extern __constant__ F cell_width;
extern __constant__ U max_num_neighbors_per_particle;

extern __constant__ U num_boundaries;
extern __constant__ F contact_tolerance;
extern __constant__ U max_num_contacts;

extern __constant__ F wrap_length;
extern __constant__ F wrap_min;
extern __constant__ F wrap_max;

void set_cubic_discretization_constants();
void set_kernel_radius(F r);
void set_particle_attr(F radius, F mass, F density);
void set_advanced_fluid_attr(F vis, F vor, F ii, F viso, F st, F stb);
void init_grid_constants(U3 res, I3 offset);
void set_cell_width(F width);
void set_search_range(F search_radius_relative_to_cell_width);
void set_max_num_particles_per_cell(U n);
void set_max_num_neighbors_per_particle(U n);
void set_gravity(F g);
void set_num_boundaries(U n);
void set_contact_tolerance(F tolerance);
void set_max_num_contacts(U n);
void set_wrap_length(F l);
}  // namespace cnst
}  // namespace alluvion

#endif
