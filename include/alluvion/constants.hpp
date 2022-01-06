#ifndef ALLUVION_CONSTANTS_HPP
#define ALLUVION_CONSTANTS_HPP

#include <iostream>
#include <limits>
#include <vector>

#include "alluvion/allocator.hpp"
#include "alluvion/data_type.hpp"
#include "alluvion/dg/peirce_quadrature.hpp"

namespace alluvion {
template <class T>
constexpr T kPi = T(
    3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651e+00L);
template <class T>
constexpr T kFMax = std::numeric_limits<T>::max();
// sqrt(3)
template <class T>
constexpr T kHcpStep0 =
    T(1.73205080756887729352744634150587236694280525381038062805581e+00L);
template <class T>
constexpr T kHcpStepInv0 =
    T(0.577350269189625764509148780501957455647601751270126876018602e+00L);
// 2 * sqrt(6) / 3
template <class T>
constexpr T kHcpStep1 =
    T(1.63299316185545206546485604980392759464396498710444675228846e+00L);
template <class T>
constexpr T kHcpStepInv1 =
    T(0.612372435695794524549321018676472847991486870164167532108173e+00L);

constexpr U kMaxNumCellsToSearch = 128;

template <U MaxNumCellsToSearch>
struct Consti {
  U max_num_particles_per_cell;
  U3 grid_res;
  I3 grid_offset;
  U max_num_neighbors_per_particle;

  U num_boundaries;
  U max_num_contacts;

  U3 hcp_grid_res;
  I3 hcp_grid_offset;
};

template <typename TF>
struct Const {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  TF kCosPhi[dg::kNumPhi];
  TF kSinPhi[dg::kNumPhi];
  TF kB[dg::kNumPhi];
  TF kR[dg::kNumR];
  TF kC[dg::kNumR];
  TF kernel_radius;
  TF kernel_radius_sqr;
  TF cubic_kernel_k;
  TF cubic_kernel_l;
  TF cubic_kernel_zero;
  TF adhesion_kernel_k;
  TF cohesion_kernel_k;
  TF cohesion_kernel_c;

  TF particle_radius;
  TF hcp_radius;
  TF hcp_radius_inv;
  TF particle_vol;
  TF particle_mass;
  TF density0;
  TF viscosity;
  TF boundary_viscosity;
  TF vorticity_coeff;
  TF boundary_vorticity_coeff;
  TF inertia_inverse;
  TF viscosity_omega;
  TF surface_tension_coeff;
  TF surface_tension_boundary_coeff;
  TF density_ghost_threshold;

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
    constexpr auto kPeirceConst = dg::PeirceConst<TF>();
    std::memcpy(kCosPhi, &kPeirceConst.cosphi, sizeof(TF) * dg::kNumPhi);
    std::memcpy(kSinPhi, &kPeirceConst.sinphi, sizeof(TF) * dg::kNumPhi);
    std::memcpy(kB, &kPeirceConst.b, sizeof(TF) * dg::kNumPhi);
    std::memcpy(kR, &kPeirceConst.r, sizeof(TF) * dg::kNumR);
    std::memcpy(kC, &kPeirceConst.c, sizeof(TF) * dg::kNumR);
  }
  void set_kernel_radius(TF h) {
    TF h2 = h * h;
    TF h3 = h2 * h;
    TF h4 = h2 * h2;
    TF tmp_cubic_k = static_cast<TF>(8.0) / h3 / kPi<TF>;
    TF tmp_cubic_l = static_cast<TF>(48.0) / h4 / kPi<TF>;
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

  void set_hcp_grid(Consti<kMaxNumCellsToSearch>& cni_arg, TF3 box_min,
                    TF3 box_max, TF radius) {
    hcp_radius = radius;
    hcp_radius_inv = 1 / radius;
    TF3 interval{radius * kHcpStep0<TF>, radius * kHcpStep1<TF>, radius * 2};
    TF3 res_fp = (box_max - box_min) / interval;
    TF3 offset_fp = box_min / interval;
    cni_arg.hcp_grid_res =
        U3{static_cast<U>(res_fp.x), static_cast<U>(res_fp.y),
           static_cast<U>(res_fp.z)};
    cni_arg.hcp_grid_offset =
        I3{static_cast<I>(offset_fp.x), static_cast<I>(offset_fp.y),
           static_cast<I>(offset_fp.z)};
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
