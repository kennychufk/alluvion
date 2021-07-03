#ifndef ALLUVION_CONSTANTS_HPP
#define ALLUVION_CONSTANTS_HPP

#include <iostream>
#include <limits>
#include <vector>

#include "alluvion/allocator.hpp"
#include "alluvion/data_type.hpp"
#include "alluvion/dg/gauss_quadrature.hpp"

namespace alluvion {
template <class T>
constexpr T kPi = T(
    3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651e+00L);
template <class T>
constexpr T kFMax = std::numeric_limits<T>::max();

constexpr U kGridN = 16;
constexpr U kGridP = 30;
constexpr U kMaxNumCellsToSearch = 128;

template <U MaxNumCellsToSearch>
struct Consti {
  U max_num_particles_per_cell;
  U3 grid_res;
  I3 grid_offset;
  U max_num_neighbors_per_particle;

  U num_boundaries;
  U max_num_contacts;
};

template <typename TF>
struct Const {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  TF kGridAbscissae[kGridN];
  TF kGridWeights[kGridN];
  TF kernel_radius;
  TF kernel_radius_sqr;
  TF cubic_kernel_k;
  TF cubic_kernel_l;
  TF cubic_kernel_zero;
  TF adhesion_kernel_k;
  TF cohesion_kernel_k;
  TF cohesion_kernel_c;

  TF particle_radius;
  TF particle_vol;
  TF particle_mass;
  TF density0;
  TF viscosity;
  TF boundary_viscosity;
  TF vorticity_coeff;
  TF inertia_inverse;
  TF viscosity_omega;
  TF surface_tension_coeff;
  TF surface_tension_boundary_coeff;

  TF3 gravity;
  TF axial_gravity;
  TF radial_gravity;

  TF boundary_epsilon;
  TF dfsph_factor_epsilon;

  TF contact_tolerance;

  TF wrap_length;
  TF wrap_min;
  TF wrap_max;

  void set_cubic_discretization_constants() {
    constexpr auto kGaussConst = dg::GaussConst<TF>();
    std::memcpy(kGridAbscissae, &kGaussConst.abscissae[kGridP][0],
                sizeof(TF) * kGridN);
    std::memcpy(kGridWeights, &kGaussConst.weights[kGridP][0],
                sizeof(TF) * kGridN);
  }
  void set_kernel_radius(TF h) {
    TF h2 = h * h;
    TF h3 = h2 * h;
    TF tmp_cubic_k = static_cast<TF>(8.0) / (kPi<TF> * h3);
    TF tmp_cubic_l = static_cast<TF>(48.0) / (kPi<TF> * h3);
    TF tmp_cubic_zero = tmp_cubic_k;
    TF tmp_adhesion_kernel_k =
        static_cast<TF>(0.007) / pow(h, static_cast<TF>(3.25));
    TF tmp_cohesion_kernel_k = static_cast<TF>(32.0) / (kPi<TF> * h3 * h3 * h3);
    TF tmp_cohesion_kernel_c = h3 * h3 / static_cast<TF>(64.0);

    kernel_radius = h;
    kernel_radius_sqr = h2;
    cubic_kernel_k = tmp_cubic_k;
    cubic_kernel_l = tmp_cubic_l;
    cubic_kernel_zero = tmp_cubic_zero;
    adhesion_kernel_k = tmp_adhesion_kernel_k;
    cohesion_kernel_k = tmp_cohesion_kernel_k;
    cohesion_kernel_c = tmp_cohesion_kernel_c;
  }

  void set_particle_attr(TF radius, TF mass, TF density) {
    TF vol = mass / density;
    particle_radius = radius;
    particle_mass = mass;
    density0 = density;
    particle_vol = vol;
  }

  void set_wrap_length(TF l) {
    TF wmin = l * static_cast<TF>(-0.5);
    TF wmax = l * static_cast<TF>(0.5);
    wrap_length = l;
    wrap_min = wmin;
    wrap_max = wmax;
  }
};
using Constf = Const<float>;
using Constd = Const<double>;
using ConstiN = Consti<kMaxNumCellsToSearch>;
extern __constant__ Constf cnf;
extern __constant__ Constd cnd;
extern __constant__ ConstiN cni;
}  // namespace alluvion

#endif
