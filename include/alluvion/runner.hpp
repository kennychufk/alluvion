#ifndef ALLUVION_RUNNER_HPP
#define ALLUVION_RUNNER_HPP
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <unordered_map>

#include "alluvion/constants.hpp"
#include "alluvion/contact.hpp"
#include "alluvion/data_type.hpp"
#include "alluvion/dg/box_distance.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/helper_math.h"
#include "alluvion/variable.hpp"
namespace alluvion {
template <class Lambda>
__device__ void forThreadMappedToElement(U element_count, Lambda f) {
  U tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < element_count) {
    f(tid);
  }
}

template <typename TF3, typename TF>
__device__ __host__ TF3 make_vector(TF x, TF y, TF z) = delete;

template <>
constexpr __device__ __host__ float3 make_vector<float3>(uint x, uint y,
                                                         uint z) {
  return float3{static_cast<float>(x), static_cast<float>(y),
                static_cast<float>(z)};
}

template <>
constexpr __device__ __host__ double3 make_vector<double3>(uint x, uint y,
                                                           uint z) {
  return double3{static_cast<double>(x), static_cast<double>(y),
                 static_cast<double>(z)};
}

template <typename TF3, typename TF>
__device__ TF3 make_vector(TF s) = delete;

template <>
inline __device__ float3 make_vector<float3>(float s) {
  return make_float3(s);
}

template <>
inline __device__ double3 make_vector<double3>(double s) {
  return make_double3(s);
}

template <typename T>
inline __host__ T from_string(std::string const& s0) = delete;

template <>
inline __host__ U from_string(std::string const& s0) {
  return stoul(s0);
}

template <>
inline __host__ float from_string(std::string const& s0) {
  return stof(s0);
}

template <typename T2>
inline __host__ T2 from_string(std::string const& s0,
                               std::string const& s1) = delete;

template <>
inline __host__ float2 from_string(std::string const& s0,
                                   std::string const& s1) {
  return float2{stof(s0), stof(s1)};
}

template <>
inline __host__ double2 from_string(std::string const& s0,
                                    std::string const& s1) {
  return double2{stod(s0), stod(s1)};
}

template <typename T3>
inline __host__ T3 from_string(std::string const& s0, std::string const& s1,
                               std::string const& s2) = delete;

template <>
inline __host__ float3 from_string(std::string const& s0, std::string const& s1,
                                   std::string const& s2) {
  return float3{stof(s0), stof(s1), stof(s2)};
}

template <>
inline __host__ double3 from_string(std::string const& s0,
                                    std::string const& s1,
                                    std::string const& s2) {
  return double3{stod(s0), stod(s1), stod(s2)};
}

template <typename TF3>
inline __device__ void get_orthogonal_vectors(TF3 vec, TF3* x, TF3* y) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  TF3 v{1, 0, 0};
  if (fabs(dot(v, vec)) > static_cast<TF>(0.999)) {
    v.x = 0;
    v.y = 1;
  }
  *x = cross(vec, v);
  *y = cross(vec, *x);
  *x = normalize(*x);
  *y = normalize(*y);
}

template <typename TQ>
inline __device__ TQ quaternion_conjugate(TQ q) {
  q.x *= -1;
  q.y *= -1;
  q.z *= -1;
  return q;
}

template <typename TQ, typename TF3>
inline __device__ __host__ TF3 rotate_using_quaternion(TF3 v, TQ q) {
  TF3 rotated;
  rotated.x = (1 - 2 * (q.y * q.y + q.z * q.z)) * v.x +
              2 * (q.x * q.y - q.z * q.w) * v.y +
              2 * (q.x * q.z + q.y * q.w) * v.z;
  rotated.y = 2 * (q.x * q.y + q.z * q.w) * v.x +
              (1 - 2 * (q.x * q.x + q.z * q.z)) * v.y +
              2 * (q.y * q.z - q.x * q.w) * v.z;
  rotated.z = 2 * (q.x * q.z - q.y * q.w) * v.x +
              2 * (q.y * q.z + q.x * q.w) * v.y +
              (1 - 2 * (q.x * q.x + q.y * q.y)) * v.z;
  return rotated;
}

template <typename TQ>
inline __device__ TQ hamilton_prod(TQ q0, TQ q1) {
  return TQ{q0.w * q1.x + q0.x * q1.w + q0.y * q1.z - q0.z * q1.y,
            q0.w * q1.y - q0.x * q1.z + q0.y * q1.w + q0.z * q1.x,
            q0.w * q1.z + q0.x * q1.y - q0.y * q1.x + q0.z * q1.w,
            q0.w * q1.w - q0.x * q1.x - q0.y * q1.y - q0.z * q1.z};
}

template <typename TQ, typename TF3>
constexpr __device__ __host__ void calculate_congruent_matrix(
    TF3 v, TQ q, TF3* congruent_diag, TF3* congruent_off_diag) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  TF qm00 = 1 - 2 * (q.y * q.y + q.z * q.z);
  TF qm01 = 2 * (q.x * q.y - q.z * q.w);
  TF qm02 = 2 * (q.x * q.z + q.y * q.w);
  TF qm10 = 2 * (q.x * q.y + q.z * q.w);
  TF qm11 = 1 - 2 * (q.x * q.x + q.z * q.z);
  TF qm12 = 2 * (q.y * q.z - q.x * q.w);
  TF qm20 = 2 * (q.x * q.z - q.y * q.w);
  TF qm21 = 2 * (q.y * q.z + q.x * q.w);
  TF qm22 = 1 - 2 * (q.x * q.x + q.y * q.y);
  congruent_diag->x = v.x * qm00 * qm00 + v.y * qm01 * qm01 + v.z * qm02 * qm02;
  congruent_diag->y = v.x * qm10 * qm10 + v.y * qm11 * qm11 + v.z * qm12 * qm12;
  congruent_diag->z = v.x * qm20 * qm20 + v.y * qm21 * qm21 + v.z * qm22 * qm22;
  congruent_off_diag->x =
      v.x * qm00 * qm10 + v.y * qm01 * qm11 + v.z * qm02 * qm12;
  congruent_off_diag->y =
      v.x * qm00 * qm20 + v.y * qm01 * qm21 + v.z * qm02 * qm22;
  congruent_off_diag->z =
      v.x * qm10 * qm20 + v.y * qm11 * qm21 + v.z * qm12 * qm22;
}

template <typename TQ, typename TF3>
constexpr __host__ TF3 calculate_angular_acceleration(TF3 inertia, TQ q,
                                                      TF3 torque) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  TF3 inertia_inverse = 1 / inertia;
  TF qm00 = 1 - 2 * (q.y * q.y + q.z * q.z);
  TF qm01 = 2 * (q.x * q.y - q.z * q.w);
  TF qm02 = 2 * (q.x * q.z + q.y * q.w);
  TF qm10 = 2 * (q.x * q.y + q.z * q.w);
  TF qm11 = 1 - 2 * (q.x * q.x + q.z * q.z);
  TF qm12 = 2 * (q.y * q.z - q.x * q.w);
  TF qm20 = 2 * (q.x * q.z - q.y * q.w);
  TF qm21 = 2 * (q.y * q.z + q.x * q.w);
  TF qm22 = 1 - 2 * (q.x * q.x + q.y * q.y);
  TF congruent0 = inertia_inverse.x * qm00 * qm00 +
                  inertia_inverse.y * qm01 * qm01 +
                  inertia_inverse.z * qm02 * qm02;
  TF congruent1 = inertia_inverse.x * qm10 * qm10 +
                  inertia_inverse.y * qm11 * qm11 +
                  inertia_inverse.z * qm12 * qm12;
  TF congruent2 = inertia_inverse.x * qm20 * qm20 +
                  inertia_inverse.y * qm21 * qm21 +
                  inertia_inverse.z * qm22 * qm22;
  TF congruent01 = inertia_inverse.x * qm00 * qm10 +
                   inertia_inverse.y * qm01 * qm11 +
                   inertia_inverse.z * qm02 * qm12;
  TF congruent02 = inertia_inverse.x * qm00 * qm20 +
                   inertia_inverse.y * qm01 * qm21 +
                   inertia_inverse.z * qm02 * qm22;
  TF congruent12 = inertia_inverse.x * qm10 * qm20 +
                   inertia_inverse.y * qm11 * qm21 +
                   inertia_inverse.z * qm12 * qm22;

  return TF3{
      torque.x * congruent0 + torque.y * congruent01 + torque.z * congruent02,
      torque.x * congruent01 + torque.y * congruent1 + torque.z * congruent12,
      torque.x * congruent02 + torque.y * congruent12 + torque.z * congruent2};
}

template <typename TQ, typename TF3>
constexpr __host__ TQ calculate_dq(TF3 omega, TQ q) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  return static_cast<TF>(0.5) *
         TQ{
             omega.x * q.w + omega.y * q.z - omega.z * q.y,   // x
             -omega.x * q.z + omega.y * q.w + omega.z * q.x,  // y
             omega.x * q.y - omega.y * q.x + omega.z * q.w,   // z
             -omega.x * q.x - omega.y * q.y - omega.z * q.z   // w
         };
}

template <typename TF3>
constexpr __device__ __host__ TF3 apply_congruent(TF3 v, TF3 congruent_diag,
                                                  TF3 congruent_off_diag) {
  return TF3{v.x * congruent_diag.x + v.y * congruent_off_diag.x +
                 v.z * congruent_off_diag.y,
             v.x * congruent_off_diag.x + v.y * congruent_diag.y +
                 v.z * congruent_off_diag.z,
             v.x * congruent_off_diag.y + v.y * congruent_off_diag.z +
                 v.z * congruent_diag.z};
}

template <typename TF3, typename TF>
constexpr __device__ __host__ void calculate_congruent_k(TF3 r, TF mass,
                                                         TF3 ii_diag,
                                                         TF3 ii_off_diag,
                                                         TF3* k_diag,
                                                         TF3* k_off_diag) {
  *k_diag = TF3{0};
  *k_off_diag = TF3{0};
  if (mass != 0) {
    TF inv_mass = 1 / mass;
    k_diag->x = r.z * r.z * ii_diag.y - r.y * r.z * ii_off_diag.z * 2 +
                r.y * r.y * ii_diag.z + inv_mass;
    k_diag->y = r.z * r.z * ii_diag.x - r.x * r.z * ii_off_diag.y * 2 +
                r.x * r.x * ii_diag.z + inv_mass;
    k_diag->z = r.y * r.y * ii_diag.x - r.x * r.y * ii_off_diag.x * 2 +
                r.x * r.x * ii_diag.y + inv_mass;

    k_off_diag->x = -r.z * r.z * ii_off_diag.x + r.x * r.z * ii_off_diag.z +
                    r.y * r.z * ii_off_diag.y - r.x * r.y * ii_diag.z;
    k_off_diag->y = r.y * r.z * ii_off_diag.x - r.x * r.z * ii_diag.y -
                    r.y * r.y * ii_off_diag.y + r.x * r.y * ii_off_diag.z;
    k_off_diag->z = -(r.y * r.z * ii_diag.x) + r.x * r.z * ii_off_diag.x +
                    r.x * r.y * ii_off_diag.y - r.x * r.x * ii_off_diag.z;
  }
}

template <typename TF3>
constexpr __device__ __host__ bool within_box(TF3 const& v, TF3 const& box_min,
                                              TF3 const& box_max) {
  return box_min.x < v.x && v.x < box_max.x && box_min.y < v.y &&
         v.y < box_max.y && box_min.z < v.z && v.z < box_max.z;
}

template <typename TQ, typename TF3>
constexpr __device__ void extract_pid(TQ const& pid_entry, TF3& x_j, U& p_j) {
  x_j = reinterpret_cast<TF3 const&>(pid_entry);
  p_j = reinterpret_cast<U const&>(pid_entry.w);
}

template <typename TQ, typename TF3>
constexpr __device__ void extract_displacement(TQ const& pid_entry,
                                               TF3& displacement) {
  displacement = reinterpret_cast<TF3 const&>(pid_entry);
}

template <typename TF>
constexpr __device__ Const<TF> const& cn() {
  if constexpr (std::is_same_v<TF, float>)
    return cnf;
  else
    return cnd;
}

template <typename TF>
constexpr __device__ TF dist_cubic_kernel(TF r) {
  TF q = r / cn<TF>().kernel_radius;
  TF conj = 1 - q;
  TF result = 0;
  if (q <= static_cast<TF>(0.5))
    result = (6 * q - 6) * q * q + 1;
  else if (q <= 1)
    result = 2 * conj * conj * conj;
  return result * cn<TF>().cubic_kernel_k;
}

template <typename TF>
constexpr __device__ TF d2_cubic_kernel(TF r2) {
  TF q2 = r2 / cn<TF>().kernel_radius_sqr;
  TF q = sqrt(r2) / cn<TF>().kernel_radius;
  TF conj = 1 - q;
  TF result = 0;
  if (q <= static_cast<TF>(0.5))
    result = (6 * q - 6) * q2 + 1;
  else if (q <= 1)
    result = 2 * conj * conj * conj;
  return result * cn<TF>().cubic_kernel_k;
}

template <typename TF3>
constexpr __device__
    std::conditional_t<std::is_same_v<TF3, float3>, float, double>
    displacement_cubic_kernel(TF3 d) {
  return d2_cubic_kernel(length_sqr(d));
}

template <typename TF>
constexpr __device__ TF d2_cubic_kernel(TF r2, TF h) {
  TF q2 = r2 / (h * h);
  TF q = sqrt(r2) / h;
  TF conj = 1 - q;
  TF result = 0;
  if (q <= static_cast<TF>(0.5))
    result = (6 * q - 6) * q2 + 1;
  else if (q <= 1)
    result = 2 * conj * conj * conj;
  return result * 8 / (kPi<TF> * h * h * h);
}

template <typename TF3, typename TF>
constexpr __device__ TF displacement_cubic_kernel(TF3 d, TF h) {
  return d2_cubic_kernel(length_sqr(d), h);
}

template <typename TF3>
inline __device__ TF3 displacement_cubic_kernel_grad(TF3 d) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  TF rl2 = length_sqr(d);
  TF rl = sqrt(rl2);
  TF q = rl / cn<TF>().kernel_radius;
  TF scale =
      (q <= static_cast<TF>(0.5) ? (q * 3 - 2) / cn<TF>().kernel_radius
                                 : (q <= 1 ? (1 - q) * (q - 1) / rl : 0));
  return cn<TF>().cubic_kernel_l * scale * d;
}

template <typename TF>
inline __device__ TF distance_adhesion_kernel(TF r2) {
  TF result = 0;
  TF r = sqrt(r2);
  if (r2 < cn<TF>().kernel_radius_sqr &&
      r > static_cast<TF>(0.5) * cn<TF>().kernel_radius) {
    // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#exponentiation-small-fractions
    result = cn<TF>().adhesion_kernel_k *
             rsqrt(rsqrt(-4 * r2 / cn<TF>().kernel_radius + 6 * r -
                         2 * cn<TF>().kernel_radius));
  }
  return result;
}

template <typename TF3>
inline __device__ std::conditional_t<std::is_same_v<TF3, float3>, float, double>
displacement_adhesion_kernel(TF3 d) {
  return distance_adhesion_kernel(length_sqr(d));
}

template <typename TF>
inline __device__ TF distance_cohesion_kernel(TF r2) {
  TF result = 0;
  TF r = sqrt(r2);
  TF r3 = r2 * r;
  TF margin = cn<TF>().kernel_radius - r;
  TF margin3 = margin * margin * margin;
  if (r2 <= cn<TF>().kernel_radius_sqr) {
    if (r > cn<TF>().kernel_radius * static_cast<TF>(0.5)) {
      result = cn<TF>().cohesion_kernel_k * margin3 * r3;
    } else {
      result = cn<TF>().cohesion_kernel_k * 2 * margin3 * r3 -
               cn<TF>().cohesion_kernel_c;
    }
  }
  return result;
}

template <typename TF3>
inline __device__ std::conditional_t<std::is_same_v<TF3, float3>, float, double>
displacement_cohesion_kernel(TF3 d) {
  return distance_cohesion_kernel(length_sqr(d));
}

template <typename TI3>
inline __device__ TI3 wrap_ipos_y(TI3 ipos) {
  if (ipos.y >= static_cast<I>(cni.grid_res.y)) {
    ipos.y -= cni.grid_res.y;
  } else if (ipos.y < 0) {
    ipos.y += cni.grid_res.y;
  }
  return ipos;
}
template <typename TF3>
inline __device__ TF3 wrap_y(TF3 v) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  if (v.y >= cn<TF>().wrap_max) {
    v.y -= cn<TF>().wrap_length;
  } else if (v.y < cn<TF>().wrap_min) {
    v.y += cn<TF>().wrap_length;
  }
  return v;
}

template <typename TF3, typename TF>
__global__ void create_fluid_cylinder_sunflower(Variable<1, TF3> particle_x,
                                                U num_particles, TF radius,
                                                U num_particles_per_slice,
                                                TF slice_distance, TF y_min) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    U slice_i = p_i / num_particles_per_slice;
    U id_in_rotation_pattern = slice_i / 4;
    U id_in_slice = p_i % num_particles_per_slice;
    TF real_in_slice = static_cast<TF>(id_in_slice) +
                       (id_in_rotation_pattern * static_cast<TF>(0.25));
    TF point_r = sqrt(real_in_slice / num_particles_per_slice) * radius;
    TF angle = kPi<TF> * (1 + sqrt(static_cast<TF>(5))) * (real_in_slice);
    particle_x(p_i) =
        TF3{point_r * cos(angle), slice_distance * slice_i + y_min,
            point_r * sin(angle)};
  });
}

template <typename TF3, typename TF>
__global__ void emit_cylinder_sunflower(Variable<1, TF3> particle_x,
                                        Variable<1, TF3> particle_v,
                                        U num_emission, U offset, TF radius,
                                        TF3 center, TF3 v) {
  forThreadMappedToElement(num_emission, [&](U i) {
    TF real_in_slice = static_cast<TF>(i) + static_cast<TF>(0.5);
    TF point_r = sqrt(real_in_slice / num_emission) * radius;
    TF angle = kPi<TF> * (1 + sqrt(static_cast<TF>(5))) * (real_in_slice);
    U p_i = i + offset;
    particle_x(p_i) =
        center + TF3{point_r * cos(angle), 0, point_r * sin(angle)};
    particle_v(p_i) = v;
  });
}

template <typename TF>
__device__ __host__ inline void get_fluid_cylinder_attr(
    TF& radius, TF y_min, TF y_max, TF particle_radius, U& sqrt_n, U& n,
    U& steps_y, TF& diameter) {
  diameter = particle_radius * 2;
  radius -= diameter;
  // sqrt_n = static_cast<U>(radius /particle_radius* sqrt(2 -
  // sqrt(static_cast<TF>(2))) ) + 1; sqrt_n -= static_cast<U>(sqrt_n % 2 == 0);
  sqrt_n = static_cast<U>(radius / particle_radius * kPi<TF> *
                          static_cast<TF>(0.25)) +
           1;
  n = sqrt_n * sqrt_n;
  steps_y =
      static_cast<I>((y_max - y_min) / diameter + static_cast<TF>(0.5)) - 1;
}

// http://l2program.co.uk/900/concentric-disk-sampling
template <typename TF3, typename TF>
__global__ void create_fluid_cylinder(Variable<1, TF3> particle_x,
                                      U num_particles, U offset, TF radius,
                                      TF y_min, TF y_max) {
  U sqrt_n;
  U n;
  U steps_y;
  TF diameter;
  get_fluid_cylinder_attr(radius, y_min, y_max, cn<TF>().particle_radius,
                          sqrt_n, n, steps_y, diameter);
  forThreadMappedToElement(num_particles, [&](U i) {
    U p_i = i + offset;

    U j = i % sqrt_n;
    U k = (i % n) / sqrt_n;
    U l = i / (n);
    TF a = static_cast<TF>(j) * 2 / (sqrt_n - 1) - 1;
    TF b = static_cast<TF>(k) * 2 / (sqrt_n - 1) - 1;
    TF r = radius;
    TF theta = 0;
    if (j == sqrt_n / 2 && j == k && sqrt_n % 2 == 1) {
      r = 0;
    } else if (a * a > b * b) {
      r *= a;
      theta = kPi<TF> * static_cast<TF>(0.25) * b / a;
    } else {
      r *= b;
      theta = kPi<TF> * static_cast<TF>(0.5) -
              kPi<TF> * static_cast<TF>(0.25) * a / b;
    }
    particle_x(p_i) =
        TF3{r * cos(theta), y_min + (l + 1) * diameter, r * sin(theta)};
  });
}

template <typename TF3>
__global__ void emit_line(Variable<1, TF3> particle_x,
                          Variable<1, TF3> particle_v, U num_emission, U offset,
                          TF3 p0, TF3 p1, TF3 v) {
  forThreadMappedToElement(num_emission, [&](U i) {
    U p_i = i + offset;
    TF3 step = (p1 - p0) / num_emission;
    particle_x(p_i) = p0 + step * i;
    particle_v(p_i) = v;
  });
}

template <typename TF3, typename TF>
__global__ void emit_if_density_lower_than_last(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF3> emission_x, Variable<1, TF> emission_sample_density,
    Variable<1, U> num_emitted, U num_emission, U offset, TF ratio_of_last,
    TF3 v) {
  forThreadMappedToElement(num_emission, [&](U i) {
    TF density_threshold =
        emission_sample_density(num_emission) * ratio_of_last;
    if (emission_sample_density(i) <= density_threshold) {
      U emit_id = atomicAdd(&num_emitted(0), 1);
      U p_i = offset + emit_id;
      particle_x(p_i) = emission_x(i);
      particle_v(p_i) = v;
    }
  });
}

template <typename TF3, typename TF>
__device__ __host__ inline void get_fluid_block_attr(
    int mode, TF3 const& box_min, TF3 const& box_max, TF particle_radius,
    I3& steps, TF& diameter, TF3& diff, TF& xshift, TF& yshift) {
  diameter = particle_radius * 2;
  TF eps = static_cast<TF>(1e-9);
  xshift = diameter;
  yshift = diameter;
  if (mode == 1) {
    yshift = sqrt(static_cast<TF>(3)) * particle_radius + eps;
  } else if (mode == 2) {
    xshift = sqrt(static_cast<TF>(6)) * diameter / 3 + eps;
    yshift = sqrt(static_cast<TF>(3)) * particle_radius + eps;
  }
  diff = box_max - box_min;
  if (mode == 1) {
    diff.x -= diameter;
    diff.z -= diameter;
  } else if (mode == 2) {
    diff.x -= xshift;
    diff.z -= diameter;
  }
  steps.x = static_cast<I>(diff.x / xshift + static_cast<TF>(0.5)) - 1;
  steps.y = static_cast<I>(diff.y / yshift + static_cast<TF>(0.5)) - 1;
  steps.z = static_cast<I>(diff.z / diameter + static_cast<TF>(0.5)) - 1;
}

template <typename TF3, typename TF>
__global__ void create_fluid_block(Variable<1, TF3> particle_x, U num_particles,
                                   U offset, int mode, TF3 box_min,
                                   TF3 box_max) {
  I3 steps;
  TF diameter;
  TF3 diff;
  TF xshift, yshift;
  get_fluid_block_attr(mode, box_min, box_max, cn<TF>().particle_radius, steps,
                       diameter, diff, xshift, yshift);
  forThreadMappedToElement(num_particles, [&](U tid) {
    U p_i = tid + offset;
    TF3 start = box_min + make_vector<TF3>(cn<TF>().particle_radius * 2);
    I j = (p_i % (steps.x * steps.z)) / steps.z;
    I k = p_i / (steps.x * steps.z);
    I l = p_i % steps.z;
    TF3 currPos = TF3{xshift * j, yshift * k, diameter * l} + start;
    TF3 shift_vec{0};
    if (mode == 1) {
      if (k % 2 == 0) {
        currPos.z += cn<TF>().particle_radius;
      } else {
        currPos.x += cn<TF>().particle_radius;
      }
    } else if (mode == 2) {
      currPos.z += cn<TF>().particle_radius;
      if (j % 2 == 1) {
        if (k % 2 == 0) {
          shift_vec.z = diameter * static_cast<TF>(0.5);
        } else {
          shift_vec.z = -diameter * static_cast<TF>(0.5);
        }
      }
      if (k % 2 == 0) {
        shift_vec.x = xshift * static_cast<TF>(0.5);
      }
    }
    particle_x(offset + p_i) = currPos + shift_vec;
  });
}

template <typename TF3>
constexpr __device__ TF3 index_to_node_position(TF3 const& domain_min,
                                                U3 const& resolution,
                                                TF3 const& cell_size, U l) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  TF3 x;
  U nv = (resolution.x + 1) * (resolution.y + 1) * (resolution.z + 1);
  U ne_x = (resolution.x + 0) * (resolution.y + 1) * (resolution.z + 1);
  U ne_y = (resolution.x + 1) * (resolution.y + 0) * (resolution.z + 1);
  U temp;
  U e_ind;

  if (l < nv) {
    temp = l % ((resolution.y + 1) * (resolution.x + 1));
    x = domain_min +
        cell_size * make_vector<TF3>(
                        temp % (resolution.x + 1), temp / (resolution.x + 1),
                        l / ((resolution.y + 1) * (resolution.x + 1)));
  } else if (l < nv + 2 * ne_x) {
    l -= nv;
    e_ind = l / 2;
    temp = e_ind % ((resolution.y + 1) * resolution.x);
    x = domain_min +
        cell_size *
            make_vector<TF3>(temp % resolution.x, temp / resolution.x,
                             e_ind / ((resolution.y + 1) * resolution.x));
    x.x += (static_cast<TF>(1) + (l % 2)) / static_cast<TF>(3) * cell_size.x;
  } else if (l < nv + 2 * (ne_x + ne_y)) {
    l -= (nv + 2 * ne_x);
    e_ind = l / 2;
    temp = e_ind % ((resolution.z + 1) * resolution.y);
    x = domain_min +
        cell_size *
            make_vector<TF3>(e_ind / ((resolution.z + 1) * resolution.y),
                             temp % resolution.y, temp / resolution.y);
    x.y += (static_cast<TF>(1) + (l % 2)) / static_cast<TF>(3) * cell_size.y;
  } else {
    l -= (nv + 2 * (ne_x + ne_y));
    e_ind = l / 2;
    temp = e_ind % ((resolution.x + 1) * resolution.z);
    x = domain_min + cell_size * make_vector<TF3>(temp / resolution.z,
                                                  e_ind / ((resolution.x + 1) *
                                                           resolution.z),
                                                  temp % resolution.z);
    x.z += (static_cast<TF>(1) + (l % 2)) / static_cast<TF>(3) * cell_size.z;
  }
  return x;
}

template <typename TF3>
__device__ void resolve(TF3 domain_min, TF3 domain_max, U3 resolution,
                        TF3 cell_size, TF3 x, I3* ipos, TF3* inner_x) {
  TF3 sd_min;
  TF3 inv_cell_size = 1 / cell_size;
  ipos->x = -1;
  if (x.x >= domain_min.x && x.y >= domain_min.y && x.z >= domain_min.z &&
      domain_max.x >= x.x && domain_max.y >= x.y && domain_max.z >= x.z) {
    *ipos = make_int3((x - domain_min) * (inv_cell_size));
    *ipos = min(*ipos, make_int3(resolution) - 1);

    sd_min = domain_min + cell_size * *ipos;

    *inner_x = 2 * (x - sd_min) * inv_cell_size - 1;
  }
}

template <typename TI3>
__device__ void get_cells(U3 resolution, TI3 ipos, U* cell) {
  U offset = (resolution.x + 1) * (resolution.y + 1) * (resolution.z + 1);
  cell[0] = (resolution.x + 1) * (resolution.y + 1) * ipos.z +
            (resolution.x + 1) * ipos.y + ipos.x;
  cell[1] = (resolution.x + 1) * (resolution.y + 1) * ipos.z +
            (resolution.x + 1) * ipos.y + ipos.x + 1;
  cell[2] = (resolution.x + 1) * (resolution.y + 1) * ipos.z +
            (resolution.x + 1) * (ipos.y + 1) + ipos.x;
  cell[3] = (resolution.x + 1) * (resolution.y + 1) * ipos.z +
            (resolution.x + 1) * (ipos.y + 1) + ipos.x + 1;
  cell[4] = (resolution.x + 1) * (resolution.y + 1) * (ipos.z + 1) +
            (resolution.x + 1) * ipos.y + ipos.x;
  cell[5] = (resolution.x + 1) * (resolution.y + 1) * (ipos.z + 1) +
            (resolution.x + 1) * ipos.y + ipos.x + 1;
  cell[6] = (resolution.x + 1) * (resolution.y + 1) * (ipos.z + 1) +
            (resolution.x + 1) * (ipos.y + 1) + ipos.x;
  cell[7] = (resolution.x + 1) * (resolution.y + 1) * (ipos.z + 1) +
            (resolution.x + 1) * (ipos.y + 1) + ipos.x + 1;

  cell[8] = offset + 2 * (resolution.x * (resolution.y + 1) * ipos.z +
                          resolution.x * ipos.y + ipos.x);
  cell[9] = cell[8] + 1;
  cell[10] = offset + 2 * (resolution.x * (resolution.y + 1) * (ipos.z + 1) +
                           resolution.x * ipos.y + ipos.x);
  cell[11] = cell[10] + 1;
  cell[12] = offset + 2 * (resolution.x * (resolution.y + 1) * ipos.z +
                           resolution.x * (ipos.y + 1) + ipos.x);
  cell[13] = cell[12] + 1;
  cell[14] = offset + 2 * (resolution.x * (resolution.y + 1) * (ipos.z + 1) +
                           resolution.x * (ipos.y + 1) + ipos.x);
  cell[15] = cell[14] + 1;

  offset += 2 * ((resolution.x + 0) * (resolution.y + 1) * (resolution.z + 1));
  cell[16] = offset + 2 * (resolution.y * (resolution.z + 1) * ipos.x +
                           resolution.y * ipos.z + ipos.y);
  cell[17] = cell[16] + 1;
  cell[18] = offset + 2 * (resolution.y * (resolution.z + 1) * (ipos.x + 1) +
                           resolution.y * ipos.z + ipos.y);
  cell[19] = cell[18] + 1;
  cell[20] = offset + 2 * (resolution.y * (resolution.z + 1) * ipos.x +
                           resolution.y * (ipos.z + 1) + ipos.y);
  cell[21] = cell[20] + 1;
  cell[22] = offset + 2 * (resolution.y * (resolution.z + 1) * (ipos.x + 1) +
                           resolution.y * (ipos.z + 1) + ipos.y);
  cell[23] = cell[22] + 1;

  offset += 2 * ((resolution.x + 1) * (resolution.y + 0) * (resolution.z + 1));
  cell[24] = offset + 2 * (resolution.z * (resolution.x + 1) * ipos.y +
                           resolution.z * ipos.x + ipos.z);
  cell[25] = cell[24] + 1;
  cell[26] = offset + 2 * (resolution.z * (resolution.x + 1) * (ipos.y + 1) +
                           resolution.z * ipos.x + ipos.z);
  cell[27] = cell[26] + 1;
  cell[28] = offset + 2 * (resolution.z * (resolution.x + 1) * ipos.y +
                           resolution.z * (ipos.x + 1) + ipos.z);
  cell[29] = cell[28] + 1;
  cell[30] = offset + 2 * (resolution.z * (resolution.x + 1) * (ipos.y + 1) +
                           resolution.z * (ipos.x + 1) + ipos.z);
  cell[31] = cell[30] + 1;
}

template <typename TF3, typename TF>
__device__ void get_shape_function(TF3 xi, TF* shape) {
  // {{{
  TF fac, fact1m3x, fact1p3x, fact1m3y, fact1p3y, fact1m3z, fact1p3z;
  TF x2 = xi.x * xi.x;
  TF y2 = xi.y * xi.y;
  TF z2 = xi.z * xi.z;

  TF _1mx = 1 - xi.x;
  TF _1my = 1 - xi.y;
  TF _1mz = 1 - xi.z;

  TF _1px = 1 + xi.x;
  TF _1py = 1 + xi.y;
  TF _1pz = 1 + xi.z;

  TF _1m3x = 1 - 3 * xi.x;
  TF _1m3y = 1 - 3 * xi.y;
  TF _1m3z = 1 - 3 * xi.z;

  TF _1p3x = 1 + 3 * xi.x;
  TF _1p3y = 1 + 3 * xi.y;
  TF _1p3z = 1 + 3 * xi.z;

  TF _1mxt1my = _1mx * _1my;
  TF _1mxt1py = _1mx * _1py;
  TF _1pxt1my = _1px * _1my;
  TF _1pxt1py = _1px * _1py;

  TF _1mxt1mz = _1mx * _1mz;
  TF _1mxt1pz = _1mx * _1pz;
  TF _1pxt1mz = _1px * _1mz;
  TF _1pxt1pz = _1px * _1pz;

  TF _1myt1mz = _1my * _1mz;
  TF _1myt1pz = _1my * _1pz;
  TF _1pyt1mz = _1py * _1mz;
  TF _1pyt1pz = _1py * _1pz;

  TF _1mx2 = 1 - x2;
  TF _1my2 = 1 - y2;
  TF _1mz2 = 1 - z2;

  constexpr TF k64th = 1 / static_cast<TF>(64);
  constexpr TF k9_64th = 9 / static_cast<TF>(64);
  // Corner nodes.
  fac = k64th * (9 * (x2 + y2 + z2) - 19);
  shape[0] = fac * _1mxt1my * _1mz;
  shape[1] = fac * _1pxt1my * _1mz;
  shape[2] = fac * _1mxt1py * _1mz;
  shape[3] = fac * _1pxt1py * _1mz;
  shape[4] = fac * _1mxt1my * _1pz;
  shape[5] = fac * _1pxt1my * _1pz;
  shape[6] = fac * _1mxt1py * _1pz;
  shape[7] = fac * _1pxt1py * _1pz;

  // Edge nodes.
  fac = k9_64th * _1mx2;
  fact1m3x = fac * _1m3x;
  fact1p3x = fac * _1p3x;
  shape[8] = fact1m3x * _1myt1mz;
  shape[9] = fact1p3x * _1myt1mz;
  shape[10] = fact1m3x * _1myt1pz;
  shape[11] = fact1p3x * _1myt1pz;
  shape[12] = fact1m3x * _1pyt1mz;
  shape[13] = fact1p3x * _1pyt1mz;
  shape[14] = fact1m3x * _1pyt1pz;
  shape[15] = fact1p3x * _1pyt1pz;

  fac = k9_64th * _1my2;
  fact1m3y = fac * _1m3y;
  fact1p3y = fac * _1p3y;
  shape[16] = fact1m3y * _1mxt1mz;
  shape[17] = fact1p3y * _1mxt1mz;
  shape[18] = fact1m3y * _1pxt1mz;
  shape[19] = fact1p3y * _1pxt1mz;
  shape[20] = fact1m3y * _1mxt1pz;
  shape[21] = fact1p3y * _1mxt1pz;
  shape[22] = fact1m3y * _1pxt1pz;
  shape[23] = fact1p3y * _1pxt1pz;

  fac = k9_64th * _1mz2;
  fact1m3z = fac * _1m3z;
  fact1p3z = fac * _1p3z;
  shape[24] = fact1m3z * _1mxt1my;
  shape[25] = fact1p3z * _1mxt1my;
  shape[26] = fact1m3z * _1mxt1py;
  shape[27] = fact1p3z * _1mxt1py;
  shape[28] = fact1m3z * _1pxt1my;
  shape[29] = fact1p3z * _1pxt1my;
  shape[30] = fact1m3z * _1pxt1py;
  shape[31] = fact1p3z * _1pxt1py;
  // }}}
}

template <typename TF3, typename TF>
__device__ void get_shape_function_and_gradient(TF3 xi, TF* shape, TF* dN0,
                                                TF* dN1, TF* dN2) {
  // {{{
  TF fac, fact1m3x, fact1p3x, fact1m3y, fact1p3y, fact1m3z, fact1p3z;
  TF x2 = xi.x * xi.x;
  TF y2 = xi.y * xi.y;
  TF z2 = xi.z * xi.z;

  TF _1mx = 1 - xi.x;
  TF _1my = 1 - xi.y;
  TF _1mz = 1 - xi.z;

  TF _1px = 1 + xi.x;
  TF _1py = 1 + xi.y;
  TF _1pz = 1 + xi.z;

  TF _1m3x = 1 - 3 * xi.x;
  TF _1m3y = 1 - 3 * xi.y;
  TF _1m3z = 1 - 3 * xi.z;

  TF _1p3x = 1 + 3 * xi.x;
  TF _1p3y = 1 + 3 * xi.y;
  TF _1p3z = 1 + 3 * xi.z;

  TF _1mxt1my = _1mx * _1my;
  TF _1mxt1py = _1mx * _1py;
  TF _1pxt1my = _1px * _1my;
  TF _1pxt1py = _1px * _1py;

  TF _1mxt1mz = _1mx * _1mz;
  TF _1mxt1pz = _1mx * _1pz;
  TF _1pxt1mz = _1px * _1mz;
  TF _1pxt1pz = _1px * _1pz;

  TF _1myt1mz = _1my * _1mz;
  TF _1myt1pz = _1my * _1pz;
  TF _1pyt1mz = _1py * _1mz;
  TF _1pyt1pz = _1py * _1pz;

  TF _1mx2 = 1 - x2;
  TF _1my2 = 1 - y2;
  TF _1mz2 = 1 - z2;
  TF _9t3x2py2pz2m19 = 9 * (3 * x2 + y2 + z2) - 19;
  TF _9tx2p3y2pz2m19 = 9 * (x2 + 3 * y2 + z2) - 19;
  TF _9tx2py2p3z2m19 = 9 * (x2 + y2 + 3 * z2) - 19;
  TF _18x = 18 * xi.x;
  TF _18y = 18 * xi.y;
  TF _18z = 18 * xi.z;

  TF _3m9x2 = 3 - 9 * x2;
  TF _3m9y2 = 3 - 9 * y2;
  TF _3m9z2 = 3 - 9 * z2;

  TF _2x = 2 * xi.x;
  TF _2y = 2 * xi.y;
  TF _2z = 2 * xi.z;

  TF _18xm9t3x2py2pz2m19 = _18x - _9t3x2py2pz2m19;
  TF _18xp9t3x2py2pz2m19 = _18x + _9t3x2py2pz2m19;
  TF _18ym9tx2p3y2pz2m19 = _18y - _9tx2p3y2pz2m19;
  TF _18yp9tx2p3y2pz2m19 = _18y + _9tx2p3y2pz2m19;
  TF _18zm9tx2py2p3z2m19 = _18z - _9tx2py2p3z2m19;
  TF _18zp9tx2py2p3z2m19 = _18z + _9tx2py2p3z2m19;

  constexpr TF k64th = 1 / static_cast<TF>(64);
  constexpr TF k9_64th = 9 / static_cast<TF>(64);
  TF fac0_8 = k64th;
  TF fac8_32 = k9_64th;
  TF _m3m9x2m2x = -_3m9x2 - _2x;
  TF _p3m9x2m2x = _3m9x2 - _2x;
  TF _1mx2t1m3x = _1mx2 * _1m3x;
  TF _1mx2t1p3x = _1mx2 * _1p3x;
  TF _m3m9y2m2y = -_3m9y2 - _2y;
  TF _p3m9y2m2y = _3m9y2 - _2y;
  TF _1my2t1m3y = _1my2 * _1m3y;
  TF _1my2t1p3y = _1my2 * _1p3y;
  TF _m3m9z2m2z = -_3m9z2 - _2z;
  TF _p3m9z2m2z = _3m9z2 - _2z;
  TF _1mz2t1m3z = _1mz2 * _1m3z;
  TF _1mz2t1p3z = _1mz2 * _1p3z;

  // Corner nodes.
  fac = k64th * (9 * (x2 + y2 + z2) - 19);
  shape[0] = fac * _1mxt1my * _1mz;
  shape[1] = fac * _1pxt1my * _1mz;
  shape[2] = fac * _1mxt1py * _1mz;
  shape[3] = fac * _1pxt1py * _1mz;
  shape[4] = fac * _1mxt1my * _1pz;
  shape[5] = fac * _1pxt1my * _1pz;
  shape[6] = fac * _1mxt1py * _1pz;
  shape[7] = fac * _1pxt1py * _1pz;

  // Edge nodes.
  fac = k9_64th * _1mx2;
  fact1m3x = fac * _1m3x;
  fact1p3x = fac * _1p3x;
  shape[8] = fact1m3x * _1myt1mz;
  shape[9] = fact1p3x * _1myt1mz;
  shape[10] = fact1m3x * _1myt1pz;
  shape[11] = fact1p3x * _1myt1pz;
  shape[12] = fact1m3x * _1pyt1mz;
  shape[13] = fact1p3x * _1pyt1mz;
  shape[14] = fact1m3x * _1pyt1pz;
  shape[15] = fact1p3x * _1pyt1pz;

  fac = k9_64th * _1my2;
  fact1m3y = fac * _1m3y;
  fact1p3y = fac * _1p3y;
  shape[16] = fact1m3y * _1mxt1mz;
  shape[17] = fact1p3y * _1mxt1mz;
  shape[18] = fact1m3y * _1pxt1mz;
  shape[19] = fact1p3y * _1pxt1mz;
  shape[20] = fact1m3y * _1mxt1pz;
  shape[21] = fact1p3y * _1mxt1pz;
  shape[22] = fact1m3y * _1pxt1pz;
  shape[23] = fact1p3y * _1pxt1pz;

  fac = k9_64th * _1mz2;
  fact1m3z = fac * _1m3z;
  fact1p3z = fac * _1p3z;
  shape[24] = fact1m3z * _1mxt1my;
  shape[25] = fact1p3z * _1mxt1my;
  shape[26] = fact1m3z * _1mxt1py;
  shape[27] = fact1p3z * _1mxt1py;
  shape[28] = fact1m3z * _1pxt1my;
  shape[29] = fact1p3z * _1pxt1my;
  shape[30] = fact1m3z * _1pxt1py;
  shape[31] = fact1p3z * _1pxt1py;

  dN0[0] = _18xm9t3x2py2pz2m19 * _1myt1mz * fac0_8;
  dN1[0] = _1mxt1mz * _18ym9tx2p3y2pz2m19 * fac0_8;
  dN2[0] = _1mxt1my * _18zm9tx2py2p3z2m19 * fac0_8;
  dN0[1] = _18xp9t3x2py2pz2m19 * _1myt1mz * fac0_8;
  dN1[1] = _1pxt1mz * _18ym9tx2p3y2pz2m19 * fac0_8;
  dN2[1] = _1pxt1my * _18zm9tx2py2p3z2m19 * fac0_8;
  dN0[2] = _18xm9t3x2py2pz2m19 * _1pyt1mz * fac0_8;
  dN1[2] = _1mxt1mz * _18yp9tx2p3y2pz2m19 * fac0_8;
  dN2[2] = _1mxt1py * _18zm9tx2py2p3z2m19 * fac0_8;
  dN0[3] = _18xp9t3x2py2pz2m19 * _1pyt1mz * fac0_8;
  dN1[3] = _1pxt1mz * _18yp9tx2p3y2pz2m19 * fac0_8;
  dN2[3] = _1pxt1py * _18zm9tx2py2p3z2m19 * fac0_8;
  dN0[4] = _18xm9t3x2py2pz2m19 * _1myt1pz * fac0_8;
  dN1[4] = _1mxt1pz * _18ym9tx2p3y2pz2m19 * fac0_8;
  dN2[4] = _1mxt1my * _18zp9tx2py2p3z2m19 * fac0_8;
  dN0[5] = _18xp9t3x2py2pz2m19 * _1myt1pz * fac0_8;
  dN1[5] = _1pxt1pz * _18ym9tx2p3y2pz2m19 * fac0_8;
  dN2[5] = _1pxt1my * _18zp9tx2py2p3z2m19 * fac0_8;
  dN0[6] = _18xm9t3x2py2pz2m19 * _1pyt1pz * fac0_8;
  dN1[6] = _1mxt1pz * _18yp9tx2p3y2pz2m19 * fac0_8;
  dN2[6] = _1mxt1py * _18zp9tx2py2p3z2m19 * fac0_8;
  dN0[7] = _18xp9t3x2py2pz2m19 * _1pyt1pz * fac0_8;
  dN1[7] = _1pxt1pz * _18yp9tx2p3y2pz2m19 * fac0_8;
  dN2[7] = _1pxt1py * _18zp9tx2py2p3z2m19 * fac0_8;

  dN0[8] = _m3m9x2m2x * _1myt1mz * fac8_32;
  dN1[8] = -_1mx2t1m3x * _1mz * fac8_32;
  dN2[8] = -_1mx2t1m3x * _1my * fac8_32;
  dN0[9] = _p3m9x2m2x * _1myt1mz * fac8_32;
  dN1[9] = -_1mx2t1p3x * _1mz * fac8_32;
  dN2[9] = -_1mx2t1p3x * _1my * fac8_32;
  dN0[10] = _m3m9x2m2x * _1myt1pz * fac8_32;
  dN1[10] = -_1mx2t1m3x * _1pz * fac8_32;
  dN2[10] = _1mx2t1m3x * _1my * fac8_32;
  dN0[11] = _p3m9x2m2x * _1myt1pz * fac8_32;
  dN1[11] = -_1mx2t1p3x * _1pz * fac8_32;
  dN2[11] = _1mx2t1p3x * _1my * fac8_32;
  dN0[12] = _m3m9x2m2x * _1pyt1mz * fac8_32;
  dN1[12] = _1mx2t1m3x * _1mz * fac8_32;
  dN2[12] = -_1mx2t1m3x * _1py * fac8_32;
  dN0[13] = _p3m9x2m2x * _1pyt1mz * fac8_32;
  dN1[13] = _1mx2t1p3x * _1mz * fac8_32;
  dN2[13] = -_1mx2t1p3x * _1py * fac8_32;
  dN0[14] = _m3m9x2m2x * _1pyt1pz * fac8_32;
  dN1[14] = _1mx2t1m3x * _1pz * fac8_32;
  dN2[14] = _1mx2t1m3x * _1py * fac8_32;
  dN0[15] = _p3m9x2m2x * _1pyt1pz * fac8_32;
  dN1[15] = _1mx2t1p3x * _1pz * fac8_32;
  dN2[15] = _1mx2t1p3x * _1py * fac8_32;

  dN0[16] = -_1my2t1m3y * _1mz * fac8_32;
  dN1[16] = _m3m9y2m2y * _1mxt1mz * fac8_32;
  dN2[16] = -_1my2t1m3y * _1mx * fac8_32;
  dN0[17] = -_1my2t1p3y * _1mz * fac8_32;
  dN1[17] = _p3m9y2m2y * _1mxt1mz * fac8_32;
  dN2[17] = -_1my2t1p3y * _1mx * fac8_32;
  dN0[18] = _1my2t1m3y * _1mz * fac8_32;
  dN1[18] = _m3m9y2m2y * _1pxt1mz * fac8_32;
  dN2[18] = -_1my2t1m3y * _1px * fac8_32;
  dN0[19] = _1my2t1p3y * _1mz * fac8_32;
  dN1[19] = _p3m9y2m2y * _1pxt1mz * fac8_32;
  dN2[19] = -_1my2t1p3y * _1px * fac8_32;
  dN0[20] = -_1my2t1m3y * _1pz * fac8_32;
  dN1[20] = _m3m9y2m2y * _1mxt1pz * fac8_32;
  dN2[20] = _1my2t1m3y * _1mx * fac8_32;
  dN0[21] = -_1my2t1p3y * _1pz * fac8_32;
  dN1[21] = _p3m9y2m2y * _1mxt1pz * fac8_32;
  dN2[21] = _1my2t1p3y * _1mx * fac8_32;
  dN0[22] = _1my2t1m3y * _1pz * fac8_32;
  dN1[22] = _m3m9y2m2y * _1pxt1pz * fac8_32;
  dN2[22] = _1my2t1m3y * _1px * fac8_32;
  dN0[23] = _1my2t1p3y * _1pz * fac8_32;
  dN1[23] = _p3m9y2m2y * _1pxt1pz * fac8_32;
  dN2[23] = _1my2t1p3y * _1px * fac8_32;

  dN0[24] = -_1mz2t1m3z * _1my * fac8_32;
  dN1[24] = -_1mz2t1m3z * _1mx * fac8_32;
  dN2[24] = _m3m9z2m2z * _1mxt1my * fac8_32;
  dN0[25] = -_1mz2t1p3z * _1my * fac8_32;
  dN1[25] = -_1mz2t1p3z * _1mx * fac8_32;
  dN2[25] = _p3m9z2m2z * _1mxt1my * fac8_32;
  dN0[26] = -_1mz2t1m3z * _1py * fac8_32;
  dN1[26] = _1mz2t1m3z * _1mx * fac8_32;
  dN2[26] = _m3m9z2m2z * _1mxt1py * fac8_32;
  dN0[27] = -_1mz2t1p3z * _1py * fac8_32;
  dN1[27] = _1mz2t1p3z * _1mx * fac8_32;
  dN2[27] = _p3m9z2m2z * _1mxt1py * fac8_32;
  dN0[28] = _1mz2t1m3z * _1my * fac8_32;
  dN1[28] = -_1mz2t1m3z * _1px * fac8_32;
  dN2[28] = _m3m9z2m2z * _1pxt1my * fac8_32;
  dN0[29] = _1mz2t1p3z * _1my * fac8_32;
  dN1[29] = -_1mz2t1p3z * _1px * fac8_32;
  dN2[29] = _p3m9z2m2z * _1pxt1my * fac8_32;
  dN0[30] = _1mz2t1m3z * _1py * fac8_32;
  dN1[30] = _1mz2t1m3z * _1px * fac8_32;
  dN2[30] = _m3m9z2m2z * _1pxt1py * fac8_32;
  dN0[31] = _1mz2t1p3z * _1py * fac8_32;
  dN1[31] = _1mz2t1p3z * _1px * fac8_32;
  dN2[31] = _p3m9z2m2z * _1pxt1py * fac8_32;
  // }}}
}

template <typename TF>
__device__ TF interpolate(Variable<1, TF>* nodes, U node_offset, U* cells,
                          TF* N) {
  TF phi = 0;
  for (U j = 0; j < 32; ++j) {
    TF c = (*nodes)(node_offset + cells[j]);
    phi += c * N[j];
  }
  return phi;
}

template <typename TF3, typename TF>
__device__ TF interpolate_distance_without_intermediates(
    Variable<1, TF>* nodes, TF3 domain_min, TF3 domain_max, U3 resolution,
    TF3 cell_size, U node_offset, TF3 x) {
  I3 ipos;
  TF3 inner_x;
  TF N[32];
  U cells[32];
  TF d = kFMax<TF>;
  resolve(domain_min, domain_max, resolution, cell_size, x, &ipos, &inner_x);
  if (ipos.x >= 0) {
    get_shape_function(inner_x, N);
    get_cells(resolution, ipos, cells);
    d = interpolate(nodes, node_offset, cells, N);
  }
  return d;
}

template <typename TF3, typename TF>
__global__ void update_volume_field(Variable<1, TF> volume_nodes,
                                    Variable<1, TF> distance_nodes,
                                    TF3 domain_min, TF3 domain_max,
                                    U3 resolution, TF3 cell_size, U num_nodes,
                                    U node_offset, TF sign, TF thickness) {
  forThreadMappedToElement(num_nodes, [&](U l) {
    TF3 x = index_to_node_position(domain_min, resolution, cell_size, l);
    TF dist = distance_nodes(node_offset + l);
    TF sum = 0;
    constexpr TF wi = 2 * kPi<TF> / dg::kNumTheta;
    for (U i = 0; i < dg::kNumTheta; ++i) {
      TF theta = 2 * kPi<TF> * (i + 1) / dg::kNumTheta;
      TF cos_theta = cos(theta);
      TF sin_theta = sin(theta);
      for (U j = 0; j < dg::kNumPhi; ++j) {
        TF3 unit_sphere_point{cn<TF>().kSinPhi[j] * cos_theta,
                              cn<TF>().kSinPhi[j] * sin_theta,
                              cn<TF>().kCosPhi[j]};
        TF wij = wi * cn<TF>().kB[j];
        for (U k = 0; k < dg::kNumR; ++k) {
          TF wijk = wij * cn<TF>().kC[k];
          TF3 integrand_parameter =
              cn<TF>().kernel_radius * cn<TF>().kR[k] * unit_sphere_point;
          TF dist_in_integrand = interpolate_distance_without_intermediates(
              &distance_nodes, domain_min, domain_max, resolution, cell_size,
              node_offset, x + integrand_parameter);
          // cubic extension function(x) =
          //   kernel(x) / kernel(0) if 0 < x < r
          //   1                     if x<= 0
          //   0                     otherwise
          // TODO: redefine
          TF dist_in_integrand_modified = dist_in_integrand * sign;
          sum += wijk * (dist_in_integrand_modified <= 0
                             ? 1
                             : dist_cubic_kernel(dist_in_integrand_modified) /
                                   cn<TF>().cubic_kernel_zero);
        }
      }
    }
    volume_nodes(node_offset + l) =
        sum * cn<TF>().kernel_radius_sqr * cn<TF>().kernel_radius;
  });
}

template <typename TF3, typename TF, typename TDistance>
__global__ void update_volume_field(Variable<1, TF> volume_nodes,
                                    const TDistance distance, TF3 domain_min,
                                    U3 resolution, TF3 cell_size, U num_nodes,
                                    U node_offset, TF sign) {
  forThreadMappedToElement(num_nodes, [&](U l) {
    TF3 x = index_to_node_position(domain_min, resolution, cell_size, l);
    TF sum = 0;
    for (U i = 0; i < dg::kNumTheta; ++i) {
      TF theta = 2 * kPi<TF> * (i + 1) / dg::kNumTheta;
      for (U j = 0; j < dg::kNumPhi; ++j) {
        for (U k = 0; k < dg::kNumR; ++k) {
          TF wijk =
              2 * kPi<TF> * cn<TF>().kB[j] * cn<TF>().kC[k] / dg::kNumTheta;
          TF3 integrand_parameter =
              cn<TF>().kernel_radius * cn<TF>().kR[k] *
              TF3{cn<TF>().kSinPhi[j] * cos(theta),
                  cn<TF>().kSinPhi[j] * sin(theta), cn<TF>().kCosPhi[j]};
          TF dist_in_integrand =
              distance.signed_distance(x + integrand_parameter);

          // cubic extension function(x) =
          //   kernel(x) / kernel(0) if 0 < x < r
          //   1                     if x<= 0
          //   0                     otherwise
          TF dist_in_integrand_modified = dist_in_integrand * sign;
          sum += wijk * (dist_in_integrand_modified <= 0
                             ? 1
                             : dist_cubic_kernel(dist_in_integrand_modified) /
                                   cn<TF>().cubic_kernel_zero);
        }
      }
    }
    volume_nodes(node_offset + l) =
        sum * cn<TF>().kernel_radius_sqr * cn<TF>().kernel_radius;
  });
}

template <typename TQ, typename TF3, typename TF>
__device__ TF compute_volume_and_boundary_x(
    Variable<1, TF>* volume_nodes, Variable<1, TF>* distance_nodes,
    TF3 domain_min, TF3 domain_max, U3 resolution, TF3 cell_size, U num_nodes,
    U node_offset, TF sign, TF thickness, TF3& x, TF3& rigid_x, TQ& rigid_q,
    TF dt, TF3* boundary_xj, TF3* xi_bxj, TF* d) {
  TF boundary_volume = 0;
  TF3 local_xi =
      rotate_using_quaternion(x - rigid_x, quaternion_conjugate(rigid_q));
  TF3 normal;
  // for resolve
  I3 ipos;
  TF3 inner_x;
  TF N[32];
  TF dN0[32];
  TF dN1[32];
  TF dN2[32];
  U cells[32];

  *boundary_xj = x;
  *xi_bxj = TF3{0};
  *d = 0;  // TODO: set to infinity?

  resolve(domain_min, domain_max, resolution, cell_size, local_xi, &ipos,
          &inner_x);
  if (ipos.x >= 0) {
    get_shape_function_and_gradient(inner_x, N, dN0, dN1, dN2);
    get_cells(resolution, ipos, cells);
    *d = interpolate_and_derive(distance_nodes, node_offset, &cell_size, cells,
                                N, dN0, dN1, dN2, &normal) *
             sign +
         thickness;
    normal = rotate_using_quaternion(normal * sign, rigid_q);
    TF nl2 = length_sqr(normal);
    normal *= (nl2 > 0 ? rsqrt(nl2) : 0);
    *xi_bxj = (*d) * normal;
    *boundary_xj = x - (*xi_bxj);
    boundary_volume = interpolate(volume_nodes, node_offset, cells, N);
  }
  return boundary_volume;
}

template <U wrap, typename TQ, typename TF3, typename TF, typename TDistance>
__device__ TF compute_volume_and_boundary_x_analytic(
    Variable<1, TF>* volume_nodes, TDistance const& distance, TF3 domain_min,
    TF3 domain_max, U3 resolution, TF3 cell_size, U num_nodes, U node_offset,
    TF sign, TF thickness, TF3& x, TF3& rigid_x, TQ& rigid_q, TF dt,
    TF3* boundary_xj, TF3* xi_bxj, TF* d) {
  TF boundary_volume = 0;
  TF3 local_xi =
      rotate_using_quaternion(x - rigid_x, quaternion_conjugate(rigid_q));
  if constexpr (wrap == 1) {
    local_xi.y = 0;
  }
  TF3 normal;
  TF nl2;
  // for grid enquiry
  I3 ipos;
  TF3 inner_x;
  TF N[32];
  U cells[32];

  *d = distance.signed_distance(local_xi) * sign + thickness;
  normal = rotate_using_quaternion(
      distance.gradient(local_xi, cn<TF>().kernel_radius) * sign, rigid_q);
  nl2 = length_sqr(normal);
  normal *= (nl2 > 0 ? rsqrt(nl2) : 0);
  *xi_bxj = (*d) * normal;
  *boundary_xj = x - (*xi_bxj);
  resolve(domain_min, domain_max, resolution, cell_size, local_xi, &ipos,
          &inner_x);
  if (ipos.x >= 0) {
    get_shape_function(inner_x, N);
    get_cells(resolution, ipos, cells);
    boundary_volume = interpolate(volume_nodes, node_offset, cells, N);
  }

  return boundary_volume;
}

// gradient must be initialized to zero
template <typename TF3, typename TF>
__device__ TF interpolate_and_derive(Variable<1, TF>* nodes, U node_offset,
                                     TF3* cell_size, U* cells, TF* N, TF* dN0,
                                     TF* dN1, TF* dN2, TF3* gradient) {
  TF phi = 0;
  *gradient = TF3{0};
  for (U j = 0; j < 32; ++j) {
    TF c = (*nodes)(node_offset + cells[j]);
    phi += c * N[j];
    gradient->x += c * dN0[j];
    gradient->y += c * dN1[j];
    gradient->z += c * dN2[j];
  }
  gradient->x *= static_cast<TF>(2) / cell_size->x;
  gradient->y *= static_cast<TF>(2) / cell_size->y;
  gradient->z *= static_cast<TF>(2) / cell_size->z;
  return phi;
}

template <typename TF3, typename TF>
__device__ TF collision_find_dist_normal(Variable<1, TF>* distance_nodes,
                                         TF3 domain_min, TF3 domain_max,
                                         U3 resolution, TF3 cell_size,
                                         U node_offset, TF sign, TF tolerance,
                                         TF3 x, TF3* normal) {
  TF dist = 0;
  I3 ipos;
  TF3 inner_x;
  TF N[32];
  TF dN0[32];
  TF dN1[32];
  TF dN2[32];
  U cells[32];
  TF d = kFMax<TF>;
  TF3 normal_tmp{0};
  *normal = TF3{0};

  resolve(domain_min, domain_max, resolution, cell_size, x, &ipos, &inner_x);
  if (ipos.x >= 0) {
    get_shape_function_and_gradient(inner_x, N, dN0, dN1, dN2);
    get_cells(resolution, ipos, cells);
    d = interpolate_and_derive(distance_nodes, node_offset, &cell_size, cells,
                               N, dN0, dN1, dN2, &normal_tmp);
  }
  if (d != kFMax<TF>) {
    dist = sign * d - tolerance;
    normal_tmp *= sign;
  }
  if (dist < 0) {
    *normal = normalize(normal_tmp);
  }
  return dist;
}

// fluid neighbor list
template <typename TF3>
__device__ I3 get_ipos(TF3 x) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  return make_int3(floor(x / cn<TF>().kernel_radius)) - cni.grid_offset;
}

template <typename TI3>
__device__ bool within_grid(TI3 ipos) {
  return 0 <= ipos.x and ipos.x < static_cast<I>(cni.grid_res.x) and
         0 <= ipos.y and ipos.y < static_cast<I>(cni.grid_res.y) and
         0 <= ipos.z and ipos.z < static_cast<I>(cni.grid_res.z);
}

template <typename TQ, typename TF3>
__global__ void update_particle_grid(Variable<1, TF3> particle_x,
                                     Variable<4, TQ> pid,
                                     Variable<3, U> pid_length,
                                     U num_particles) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    I3 ipos = get_ipos(x_i);
    U pid_insert_index;
    if (within_grid(ipos)) {
      pid_insert_index = atomicAdd(&pid_length(ipos), 1);
      if (pid_insert_index == cni.max_num_particles_per_cell) {
        printf("Too many particles at ipos = (%d, %d, %d)\n", ipos.x, ipos.y,
               ipos.z);
      }
      // NOTE: pack x_i and p_i together to avoid scatter-read from particle_x
      // in make_neighbor_list
      TQ pid_entry{x_i.x, x_i.y, x_i.z, 0};
      reinterpret_cast<U&>(pid_entry.w) = p_i;
      pid(ipos, pid_insert_index) = pid_entry;
    } else {
      printf("Particle falls out of the grid\n");
    }
  });
}

template <U wrap, typename TQ, typename TF3>
__global__ void make_neighbor_list(Variable<1, TF3> sample_x,
                                   Variable<4, TQ> pid,
                                   Variable<3, U> pid_length,
                                   Variable<2, TQ> sample_neighbors,
                                   Variable<1, U> sample_num_neighbors,
                                   U num_samples) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF3 x = sample_x(p_i);
    I3 ipos = get_ipos(x);
    U num_neighbors = 0;
#pragma unroll
    for (I i = 0; i < 27; ++i) {
      I3 neighbor_ipos = ipos + I3{i / 9, (i / 3) % 3, i % 3} - 1;
      if constexpr (wrap == 1) neighbor_ipos = wrap_ipos_y(neighbor_ipos);
      if (within_grid(neighbor_ipos)) {
        U neighbor_occupancy =
            min(pid_length(neighbor_ipos), cni.max_num_particles_per_cell);
        for (U k = 0; k < neighbor_occupancy; ++k) {
          TQ pid_entry = pid(neighbor_ipos, k);
          TF3 x_j;
          U p_j;
          extract_pid(pid_entry, x_j, p_j);
          TF3 xixj = x - x_j;
          if constexpr (wrap == 1) xixj = wrap_y(xixj);
          if (p_j != p_i && length_sqr(xixj) < cn<TF>().kernel_radius_sqr) {
            TQ neighbor_entry{xixj.x, xixj.y, xixj.z, 0};
            reinterpret_cast<U&>(neighbor_entry.w) = p_j;
            sample_neighbors(p_i, num_neighbors) = neighbor_entry;
            num_neighbors += 1;
            if (num_neighbors == cni.max_num_neighbors_per_particle) {
              printf("Particle %d has too many neighbors\n", p_i);
            }
          }
        }
      }
    }
    sample_num_neighbors(p_i) = num_neighbors;
  });
}

// fluid
template <typename TF3>
__global__ void clear_acceleration(Variable<1, TF3> particle_a,
                                   U num_particles) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  forThreadMappedToElement(num_particles,
                           [&](U p_i) { particle_a(p_i) = cn<TF>().gravity; });
}

template <typename TF3>
__global__ void apply_axial_gravitation(Variable<1, TF3> particle_a,
                                        Variable<1, TF3> particle_x,
                                        U num_particles) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    particle_a(p_i) =
        TF3{x_i.x * cn<TF>().radial_gravity, x_i.y * cn<TF>().axial_gravity,
            x_i.z * cn<TF>().radial_gravity};
  });
}

template <U wrap, typename TQ, typename TF3, typename TF, typename TDistance>
__global__ void compute_particle_boundary_analytic(
    Variable<1, TF> volume_nodes, const TDistance distance, TF3 rigid_x,
    TQ rigid_q, U boundary_id, TF3 domain_min, TF3 domain_max, U3 resolution,
    TF3 cell_size, U num_nodes, U node_offset, TF sign, TF thickness, TF dt,
    Variable<1, TF3> particle_x, Variable<2, TQ> particle_boundary,
    Variable<2, TQ> particle_boundary_kernel, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 boundary_xj, xi_bxj;
    TF d;
    TF boundary_volume = compute_volume_and_boundary_x_analytic<wrap>(
        &volume_nodes, distance, domain_min, domain_max, resolution, cell_size,
        num_nodes, node_offset, sign, thickness, particle_x(p_i), rigid_x,
        rigid_q, dt, &boundary_xj, &xi_bxj, &d);
    TF3 kernel_grad = displacement_cubic_kernel_grad(xi_bxj);
    particle_boundary(boundary_id, p_i) =
        TQ{boundary_xj.x, boundary_xj.y, boundary_xj.z, boundary_volume};
    particle_boundary_kernel(boundary_id, p_i) =
        boundary_volume *
        TQ{kernel_grad.x, kernel_grad.y, kernel_grad.z, dist_cubic_kernel(d)};
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_particle_boundary(
    Variable<1, TF> volume_nodes, Variable<1, TF> distance_nodes, TF3 rigid_x,
    TQ rigid_q, U boundary_id, TF3 domain_min, TF3 domain_max, U3 resolution,
    TF3 cell_size, U num_nodes, U node_offset, TF sign, TF thickness, TF dt,
    Variable<1, TF3> particle_x, Variable<2, TQ> particle_boundary,
    Variable<2, TQ> particle_boundary_kernel, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 boundary_xj, xi_bxj;
    TF d;
    TF boundary_volume = compute_volume_and_boundary_x(
        &volume_nodes, &distance_nodes, domain_min, domain_max, resolution,
        cell_size, num_nodes, node_offset, sign, thickness, particle_x(p_i),
        rigid_x, rigid_q, dt, &boundary_xj, &xi_bxj, &d);
    TF3 kernel_grad = displacement_cubic_kernel_grad(xi_bxj);
    particle_boundary(boundary_id, p_i) =
        TQ{boundary_xj.x, boundary_xj.y, boundary_xj.z, boundary_volume};
    particle_boundary_kernel(boundary_id, p_i) =
        boundary_volume *
        TQ{kernel_grad.x, kernel_grad.y, kernel_grad.z, dist_cubic_kernel(d)};
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_density(Variable<1, TF3> particle_x,
                                Variable<2, TQ> particle_neighbors,
                                Variable<1, U> particle_num_neighbors,
                                Variable<1, TF> particle_density,
                                Variable<2, TQ> particle_boundary_kernel,
                                U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF density = cn<TF>().particle_vol * cn<TF>().cubic_kernel_zero;
    TF3 x_i = particle_x(p_i);
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      extract_displacement(particle_neighbors(p_i, neighbor_id), xixj);
      density += cn<TF>().particle_vol * displacement_cubic_kernel(xixj);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      density += particle_boundary_kernel(boundary_id, p_i).w;
    }
    particle_density(p_i) = density * cn<TF>().density0;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void sample_position_density(Variable<1, TF3> sample_x,
                                        Variable<2, TQ> sample_neighbors,
                                        Variable<1, U> sample_num_neighbors,
                                        Variable<1, TF> sample_density,
                                        Variable<2, TQ> sample_boundary_kernel,
                                        U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF density = 0;
    TF3 x_i = sample_x(p_i);
    for (U neighbor_id = 0; neighbor_id < sample_num_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      extract_displacement(sample_neighbors(p_i, neighbor_id), xixj);
      density += cn<TF>().particle_vol * displacement_cubic_kernel(xixj);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      density += sample_boundary_kernel(boundary_id, p_i).w;
    }
    sample_density(p_i) = density * cn<TF>().density0;
  });
}

// Viscosity_Standard
template <typename TQ, typename TF3, typename TF>
__global__ void compute_viscosity(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, Variable<1, TF3> particle_a,
    Variable<2, TF3> particle_force, Variable<2, TF3> particle_torque,
    Variable<2, TQ> particle_boundary, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 v_i = particle_v(p_i);
    TF3 x_i = particle_x(p_i);
    TF3 da{0};

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      da += cn<TF>().viscosity * cn<TF>().particle_mass /
            particle_density(p_j) *
            dot(xixj, displacement_cubic_kernel_grad(xixj)) /
            (length_sqr(xixj) +
             static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
            (v_i - particle_v(p_j));
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = particle_boundary(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
      TF3 r_x = rigid_x(boundary_id);
      TF3 r_v = rigid_v(boundary_id);
      TF3 r_omega = rigid_omega(boundary_id);

      TF3 normal = bx_j - x_i;
      TF nl2 = length_sqr(normal);
      if (nl2 > 0) {
        normal *= rsqrt(nl2);
        TF3 t1, t2;
        get_orthogonal_vectors(normal, &t1, &t2);

        TF dist = cn<TF>().kernel_radius - sqrt(nl2);
        TF3 x1 = bx_j - t1 * dist;
        TF3 x2 = bx_j + t1 * dist;
        TF3 x3 = bx_j - t2 * dist;
        TF3 x4 = bx_j + t2 * dist;

        TF3 xix1 = x_i - x1;
        TF3 xix2 = x_i - x2;
        TF3 xix3 = x_i - x3;
        TF3 xix4 = x_i - x4;

        TF3 gradW1 = displacement_cubic_kernel_grad(xix1);
        TF3 gradW2 = displacement_cubic_kernel_grad(xix2);
        TF3 gradW3 = displacement_cubic_kernel_grad(xix3);
        TF3 gradW4 = displacement_cubic_kernel_grad(xix4);

        // each sample point represents the quarter of the volume inside of the
        // boundary
        TF vol = static_cast<TF>(0.25) * boundary.w;

        TF3 v1 = cross(r_omega, x1 - r_x) + r_v;
        TF3 v2 = cross(r_omega, x2 - r_x) + r_v;
        TF3 v3 = cross(r_omega, x3 - r_x) + r_v;
        TF3 v4 = cross(r_omega, x4 - r_x) + r_v;

        // compute forces for both sample point
        TF3 a1 = cn<TF>().boundary_viscosity * vol * dot(xix1, gradW1) /
                 (length_sqr(xix1) +
                  static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
                 (v_i - v1);
        TF3 a2 = cn<TF>().boundary_viscosity * vol * dot(xix2, gradW2) /
                 (length_sqr(xix2) +
                  static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
                 (v_i - v2);
        TF3 a3 = cn<TF>().boundary_viscosity * vol * dot(xix3, gradW3) /
                 (length_sqr(xix3) +
                  static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
                 (v_i - v3);
        TF3 a4 = cn<TF>().boundary_viscosity * vol * dot(xix4, gradW4) /
                 (length_sqr(xix4) +
                  static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
                 (v_i - v4);
        da += a1 + a2 + a3 + a4;

        TF3 f1 = -cn<TF>().particle_mass * a1;
        TF3 f2 = -cn<TF>().particle_mass * a2;
        TF3 f3 = -cn<TF>().particle_mass * a3;
        TF3 f4 = -cn<TF>().particle_mass * a4;
        TF3 torque1 = cross(x1 - r_x, f1);
        TF3 torque2 = cross(x2 - r_x, f2);
        TF3 torque3 = cross(x3 - r_x, f3);
        TF3 torque4 = cross(x4 - r_x, f4);

        particle_force(boundary_id, p_i) += f1 + f2 + f3 + f4;
        particle_torque(boundary_id, p_i) +=
            torque1 + torque2 + torque3 + torque4;
      }
    }
    particle_a(p_i) += da;
  });
}

// Micropolar Model
template <typename TQ, typename TF3, typename TF>
__global__ void compute_vorticity(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<1, TF3> particle_omega,
    Variable<1, TF3> particle_a, Variable<1, TF3> particle_angular_acceleration,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TF3> particle_force, Variable<2, TF3> particle_torque,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    Variable<1, TF3> rigid_x, Variable<1, TF3> rigid_v,
    Variable<1, TF3> rigid_omega, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v_i = particle_v(p_i);
    TF3 omegai = particle_omega(p_i);
    TF density_i = particle_density(p_i);

    TF3 da{0};
    TF3 dangular_acc =
        -2 * cn<TF>().inertia_inverse * cn<TF>().vorticity_coeff * omegai;

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      TF3 omegaij = omegai - particle_omega(p_j);
      TF3 grad_w = displacement_cubic_kernel_grad(xixj);

      dangular_acc -= cn<TF>().inertia_inverse * cn<TF>().viscosity_omega / dt *
                      cn<TF>().particle_mass / particle_density(p_j) * omegaij *
                      displacement_cubic_kernel(xixj);
      da += cn<TF>().vorticity_coeff / density_i * cn<TF>().particle_mass *
            cross(omegaij, grad_w);
      dangular_acc +=
          cn<TF>().vorticity_coeff / density_i * cn<TF>().inertia_inverse *
          cross(cn<TF>().particle_mass * (v_i - particle_v(p_j)), grad_w);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = particle_boundary(boundary_id, p_i);
      TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
      TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
                 rigid_v(boundary_id);
      TF3 omegaij = omegai;  // TODO: omegaj not implemented in SPlisHSPlasH
      TF3 a = cn<TF>().vorticity_coeff / density_i * cn<TF>().density0 *
              cross(omegaij, grad_wvol);
      da += a;
      dangular_acc += cn<TF>().vorticity_coeff / density_i *
                      cn<TF>().inertia_inverse * cn<TF>().density0 *
                      cross(v_i - velj, grad_wvol);
      TF3 force = -cn<TF>().particle_mass * a;
      particle_force(boundary_id, p_i) += force;
      particle_torque(boundary_id, p_i) +=
          cross(bx_j - rigid_x(boundary_id), force);
    }

    particle_a(p_i) += da;
    particle_angular_acceleration(p_i) = dangular_acc;
  });
}
template <typename TF3, typename TF>
__global__ void integrate_angular_acceleration(
    Variable<1, TF3> particle_omega,
    Variable<1, TF3> particle_angular_acceleration, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_omega(p_i) += dt * particle_angular_acceleration(p_i);
  });
}
template <typename TQ, typename TF3, typename TF>
__global__ void compute_normal(Variable<1, TF3> particle_x,
                               Variable<1, TF> particle_density,
                               Variable<1, TF3> particle_normal,
                               Variable<2, TQ> particle_neighbors,
                               Variable<1, U> particle_num_neighbors,
                               U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 ni{0};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      ni += cn<TF>().particle_mass / particle_density(p_j) *
            displacement_cubic_kernel_grad(xixj);
    }
    particle_normal(p_i) = ni;
  });
}
template <typename TQ, typename TF3, typename TF>
__global__ void compute_surface_tension(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF3> particle_normal, Variable<1, TF3> particle_a,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 ni = particle_normal(p_i);
    TF density_i = particle_density(p_i);

    TF3 da{0};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      TF k_ij = cn<TF>().density0 * 2 / (density_i + particle_density(p_j));
      TF length2 = length_sqr(xixj);
      TF3 accel{0};
      if (length2 > static_cast<TF>(1e-9)) {
        accel = -cn<TF>().surface_tension_coeff * cn<TF>().particle_mass *
                displacement_cohesion_kernel(xixj) * rsqrt(length2) * xixj;
      }
      accel -= cn<TF>().surface_tension_coeff * (ni - particle_normal(p_j));
      da += k_ij * accel;
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = particle_boundary(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
      TF3 xixj = x_i - bx_j;
      TF length2 = length_sqr(xixj);
      if (length2 > 0) {
        da -= cn<TF>().surface_tension_boundary_coeff * boundary.w *
              cn<TF>().density0 * displacement_adhesion_kernel(xixj) *
              rsqrt(length2) * xixj;
      }
    }
    particle_a(p_i) += da;
  });
}

template <typename TF3, typename TF>
__global__ void calculate_cfl_v2(Variable<1, TF3> particle_v,
                                 Variable<1, TF3> particle_a,
                                 Variable<1, TF> particle_cfl_v2, TF dt,
                                 U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_cfl_v2(p_i) = length_sqr(particle_v(p_i) + particle_a(p_i) * dt);
  });
}
template <typename TQ, typename TF3, typename TF>
__global__ void predict_advection0(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF3> particle_a, Variable<1, TF> particle_density,
    Variable<1, TF3> particle_dii, Variable<1, TF> particle_pressure,
    Variable<1, TF> particle_last_pressure, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary_kernel, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_v(p_i) += dt * particle_a(p_i);
    particle_last_pressure(p_i) = particle_pressure(p_i) * static_cast<TF>(0.5);
    TF3 x_i = particle_x(p_i);
    TF density = particle_density(p_i);
    TF density2 = density * density;

    TF3 dii{0};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      extract_displacement(particle_neighbors(p_i, neighbor_id), xixj);
      dii -= cn<TF>().particle_vol / density2 *
             displacement_cubic_kernel_grad(xixj);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
      dii -= grad_wvol / density2;
    }
    particle_dii(p_i) = dii * cn<TF>().density0;
  });
}
template <typename TQ, typename TF3, typename TF>
__global__ void predict_advection1(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF3> particle_dii, Variable<1, TF> particle_adv_density,
    Variable<1, TF> particle_aii, Variable<1, TF> particle_density,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    Variable<1, TF3> rigid_x, Variable<1, TF3> rigid_v,
    Variable<1, TF3> rigid_omega, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v = particle_v(p_i);
    TF3 dii = particle_dii(p_i);
    TF density = particle_density(p_i);
    TF density2 = density * density;
    TF inv_density2 = 1 / density2;
    TF dpi = cn<TF>().particle_vol * inv_density2;

    // target
    TF density_adv = density / cn<TF>().density0;
    TF aii = 0;

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);

      TF3 grad_w = displacement_cubic_kernel_grad(xixj);
      TF3 dji = dpi * cn<TF>().density0 * grad_w;

      density_adv +=
          dt * cn<TF>().particle_vol * dot(v - particle_v(p_j), grad_w);
      aii += cn<TF>().particle_vol * dot(dii - dji, grad_w);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = particle_boundary(boundary_id, p_i);
      TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);

      TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
                 rigid_v(boundary_id);
      TF3 dji = inv_density2 * cn<TF>().density0 * grad_wvol;
      density_adv += dt * dot(v - velj, grad_wvol);
      aii += dot(dii - dji, grad_wvol);
    }
    particle_adv_density(p_i) = density_adv;
    particle_aii(p_i) = aii * cn<TF>().density0;
  });
}
template <typename TQ, typename TF3, typename TF>
__global__ void pressure_solve_iteration0(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_last_pressure, Variable<1, TF3> particle_dij_pj,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    // target
    TF3 dij_pj{0};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);

      TF densityj = particle_density(p_j);
      TF densityj2 = densityj * densityj;
      TF last_pressure = particle_last_pressure(p_j);

      dij_pj -= cn<TF>().particle_vol / densityj2 * last_pressure *
                displacement_cubic_kernel_grad(xixj);
    }
    particle_dij_pj(p_i) = dij_pj * cn<TF>().density0;
  });
}
template <typename TQ, typename TF3, typename TF>
__global__ void pressure_solve_iteration1(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_last_pressure, Variable<1, TF> particle_pressure,
    Variable<1, TF3> particle_dii, Variable<1, TF3> particle_dij_pj,
    Variable<1, TF> particle_aii, Variable<1, TF> particle_adv_density,
    Variable<1, TF> particle_density_err, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary_kernel, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF last_pressure = particle_last_pressure(p_i);
    TF3 dij_pj = particle_dij_pj(p_i);
    TF density = particle_density(p_i);
    TF density2 = density * density;
    TF dpi = cn<TF>().particle_vol / density2;

    TF b = 1 - particle_adv_density(p_i);
    TF dt2 = dt * dt;
    TF aii = particle_aii(p_i);
    TF denom = aii * dt2;
    constexpr TF jacobi_weight = static_cast<TF>(2) / 3;
    TF pressure = 0;

    TF sum_tmp = 0;
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);

      TF3 djk_pk = particle_dij_pj(p_j);
      TF3 grad_w = displacement_cubic_kernel_grad(xixj);
      TF3 dji = dpi * cn<TF>().density0 * grad_w;
      TF3 dji_pi = dji * last_pressure;

      sum_tmp += cn<TF>().particle_vol *
                 dot(dij_pj - particle_dii(p_j) * particle_last_pressure(p_j) -
                         (djk_pk - dji_pi),
                     grad_w);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
      sum_tmp += dot(dij_pj, grad_wvol);
    }
    sum_tmp *= cn<TF>().density0;
    if (fabs(denom) > static_cast<TF>(1.0e-9)) {
      pressure = max((1 - jacobi_weight) * last_pressure +
                         jacobi_weight / denom * (b - dt2 * sum_tmp),
                     static_cast<TF>(0));
    }
    particle_density_err(p_i) =
        pressure == 0 ? 0 : (denom * pressure + sum_tmp * dt2 - b);
    particle_pressure(p_i) = pressure;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_pressure_accels(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_pressure, Variable<1, TF3> particle_pressure_accel,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TF3> particle_force, Variable<2, TF3> particle_torque,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    Variable<1, TF3> rigid_x, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF density = particle_density(p_i);
    TF density2 = density * density;
    TF dpi = particle_pressure(p_i) / density2;

    // target
    TF3 ai{0};

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);

      TF densityj = particle_density(p_j);
      TF densityj2 = densityj * densityj;
      TF dpj = particle_pressure(p_j) / densityj2;
      ai -= cn<TF>().particle_vol * (dpi + dpj) *
            displacement_cubic_kernel_grad(xixj);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = particle_boundary(boundary_id, p_i);
      TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
      TF3 a = dpi * grad_wvol;
      TF3 force = cn<TF>().particle_mass * a;
      ai -= a;
      particle_force(boundary_id, p_i) += force;
      particle_torque(boundary_id, p_i) +=
          cross(bx_j - rigid_x(boundary_id), force);
    }
    particle_pressure_accel(p_i) = cn<TF>().density0 * cn<TF>().density0 * ai;
  });
}
template <U wrap, typename TF3, typename TF>
__global__ void kinematic_integration(Variable<1, TF3> particle_x,
                                      Variable<1, TF3> particle_v,
                                      Variable<1, TF3> particle_pressure_accel,
                                      TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 v = particle_v(p_i);
    v += particle_pressure_accel(p_i) * dt;
    if constexpr (wrap == 0)
      particle_x(p_i) += v * dt;
    else
      particle_x(p_i) = wrap_y(particle_x(p_i) + v * dt);
    particle_v(p_i) = v;
  });
}
template <typename TF3, typename TF>
__global__ void move_particles(Variable<1, TF3> particle_x,
                               Variable<1, TF3> particle_v, TF dt,
                               TF3 exclusion_min, TF3 exclusion_max,
                               U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    const TF3 x_i = particle_x(p_i);
    if (x_i.x < exclusion_min.x || x_i.x > exclusion_max.x ||
        x_i.y < exclusion_min.y || x_i.y > exclusion_max.y ||
        x_i.z < exclusion_min.z || x_i.z > exclusion_max.z) {
      particle_x(p_i) = x_i + dt * particle_v(p_i);
    }
  });
}

// ISPH
template <typename TF3, typename TF>
__global__ void advect_and_init_pressure(Variable<1, TF3> particle_v,
                                         Variable<1, TF3> particle_a,
                                         Variable<1, TF> particle_pressure,
                                         Variable<1, TF> particle_last_pressure,
                                         TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_v(p_i) += dt * particle_a(p_i);
    particle_last_pressure(p_i) = particle_pressure(p_i) * static_cast<TF>(0.5);
  });
}
template <typename TQ, typename TF3, typename TF2, typename TF>
__global__ void calculate_isph_diagonal_adv_density(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density,
    Variable<1, TF2> particle_diag_adv_density,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    Variable<1, TF3> rigid_x, Variable<1, TF3> rigid_v,
    Variable<1, TF3> rigid_omega, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v = particle_v(p_i);
    TF density = particle_density(p_i);

    // target
    TF density_adv = density / cn<TF>().density0;
    TF diag = 0;

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      TF densityj = particle_density(p_j);

      TF3 grad_w = displacement_cubic_kernel_grad(xixj);

      density_adv +=
          dt * cn<TF>().particle_vol * dot(v - particle_v(p_j), grad_w);
      diag += cn<TF>().particle_vol * (density + densityj) /
              (density * densityj) /
              (length_sqr(xixj) +
               static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
              dot(xixj, grad_w);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = particle_boundary(boundary_id, p_i);
      TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);

      TF3 xib = x_i - bx_j;
      TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
                 rigid_v(boundary_id);
      density_adv += dt * dot(v - velj, grad_wvol);
      diag += (density + cn<TF>().density0) / (density * cn<TF>().density0) /
              (length_sqr(xib) +
               static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
              dot(xib, grad_wvol);
    }
    particle_diag_adv_density(p_i) = TF2{diag * cn<TF>().density0, density_adv};
  });
}

template <typename TQ, typename TF3, typename TF2, typename TF>
__global__ void isph_solve_iteration(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_last_pressure, Variable<1, TF> particle_pressure,
    Variable<1, TF2> particle_diag_adv_density,
    Variable<1, TF> particle_density_err, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF last_pressure = particle_last_pressure(p_i);
    TF density = particle_density(p_i);
    TF2 diag_adv_density = particle_diag_adv_density(p_i);

    TF b = 1 - diag_adv_density.y;
    TF dt2 = dt * dt;
    TF denom = diag_adv_density.x * dt2;
    // https://ocw.mit.edu/courses/mathematics/18-086-mathematical-methods-for-engineers-ii-spring-2006/readings/am62.pdf
    constexpr TF jacobi_weight = static_cast<TF>(2) / 3;
    TF pressure = 0;

    TF off_diag = 0;
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);

      TF densityj = particle_density(p_j);
      TF3 grad_w = displacement_cubic_kernel_grad(xixj);

      off_diag -= cn<TF>().particle_vol * (density + densityj) /
                  (density * densityj) /
                  (length_sqr(xixj) +
                   static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
                  dot(xixj, grad_w) * particle_last_pressure(p_j);
    }
    off_diag *= cn<TF>().density0;
    if (fabs(denom) > static_cast<TF>(1.0e-9)) {
      pressure = max((1 - jacobi_weight) * last_pressure +
                         jacobi_weight / denom * (b - dt2 * off_diag),
                     static_cast<TF>(0));
    }
    particle_density_err(p_i) =
        pressure == 0 ? 0 : (denom * pressure + off_diag * dt2 - b);
    particle_pressure(p_i) = pressure;
  });
}

// DFSPH
template <typename TQ, typename TF3, typename TF>
__global__ void compute_dfsph_factor(Variable<1, TF3> particle_x,
                                     Variable<2, TQ> particle_neighbors,
                                     Variable<1, U> particle_num_neighbors,
                                     Variable<1, TF> particle_dfsph_factor,
                                     Variable<2, TQ> particle_boundary_kernel,
                                     U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF sum_grad_p_k = 0;
    TF3 grad_p_i{};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      extract_displacement(particle_neighbors(p_i, neighbor_id), xixj);
      TF3 grad_p_j =
          -cn<TF>().particle_vol * displacement_cubic_kernel_grad(xixj);
      sum_grad_p_k += length_sqr(grad_p_j);
      grad_p_i -= grad_p_j;
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
      grad_p_i += grad_wvol;
    }
    sum_grad_p_k += length_sqr(grad_p_i);
    particle_dfsph_factor(p_i) = sum_grad_p_k > cn<TF>().dfsph_factor_epsilon
                                     ? -1 / sum_grad_p_k
                                     : static_cast<TF>(0);  // TODO: necessary?
  });
}

template <typename TQ, typename TF3, typename TF>
__device__ TF compute_density_change(
    Variable<1, TF3>& particle_x, Variable<1, TF3>& particle_v,
    Variable<2, TQ>& particle_neighbors, Variable<1, U>& particle_num_neighbors,
    Variable<2, TQ>& particle_boundary,
    Variable<2, TQ>& particle_boundary_kernel, Variable<1, TF3>& rigid_x,
    Variable<1, TF3>& rigid_v, Variable<1, TF3>& rigid_omega, U p_i) {
  TF density_adv = 0;
  TF3 x_i = particle_x(p_i);
  TF3 v_i = particle_v(p_i);
  for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
       ++neighbor_id) {
    U p_j;
    TF3 xixj;
    extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
    density_adv +=
        cn<TF>().particle_vol *
        dot(v_i - particle_v(p_j), displacement_cubic_kernel_grad(xixj));
  }

  for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
    TQ boundary = particle_boundary(boundary_id, p_i);
    TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
    TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
    TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
    TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
               rigid_v(boundary_id);
    density_adv += dot(v_i - velj, grad_wvol);
  }
  density_adv = max(density_adv, static_cast<TF>(0));
  if (particle_num_neighbors(p_i) <
      20) {  // TODO: 20 as configurable constant; duplicate reading of
             // particle_num_neighbors
    density_adv = 0;
  }
  return density_adv;
}

template <typename TQ, typename TF3, typename TF>
__global__ void warm_start_divergence_solve_0(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_kappa_v, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, Variable<2, TQ> particle_boundary,
    Variable<2, TQ> particle_boundary_kernel, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, TF dt,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF density_adv = compute_density_change<TQ, TF3, TF>(
        particle_x, particle_v, particle_neighbors, particle_num_neighbors,
        particle_boundary, particle_boundary_kernel, rigid_x, rigid_v,
        rigid_omega, p_i);
    particle_kappa_v(p_i) =
        (density_adv > 0)
            ? static_cast<TF>(0.5) *
                  max(particle_kappa_v(p_i), static_cast<TF>(-0.5)) / dt
            : static_cast<TF>(0);
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void warm_start_divergence_solve_1(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_kappa_v, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, Variable<2, TQ> particle_boundary,
    Variable<2, TQ> particle_boundary_kernel, Variable<2, TF3> particle_force,
    Variable<2, TF3> particle_torque, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, TF dt,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v_i = particle_v(p_i);
    TF k_i = particle_kappa_v(p_i);
    TF3 dv{};

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);

      TF k_sum = (k_i + particle_kappa_v(p_j));
      if (fabs(k_sum) > cn<TF>().dfsph_factor_epsilon) {  // TODO: new epsilon?

        TF3 grad_p_j =
            -cn<TF>().particle_vol * displacement_cubic_kernel_grad(xixj);
        dv -= dt * k_sum * grad_p_j;  // ki, kj already contain inverse density
      }
    }
    if (fabs(k_i) > cn<TF>().dfsph_factor_epsilon) {  // TODO: new epsilon?
      for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
        TQ boundary = particle_boundary(boundary_id, p_i);
        TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
        TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
        TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
        TF3 a = k_i * -grad_wvol;
        TF3 force = cn<TF>().particle_mass * a;
        dv -= dt * a;
        particle_force(boundary_id, p_i) += force;
        particle_torque(boundary_id, p_i) +=
            cross(bx_j - rigid_x(boundary_id), force);
      }
    }
    particle_v(p_i) += dv;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_velocity_of_density_change(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_dfsph_factor, Variable<1, TF> particle_density_adv,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    Variable<1, TF3> rigid_x, Variable<1, TF3> rigid_v,
    Variable<1, TF3> rigid_omega, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_density_adv(p_i) = compute_density_change<TQ, TF3, TF>(
        particle_x, particle_v, particle_neighbors, particle_num_neighbors,
        particle_boundary, particle_boundary_kernel, rigid_x, rigid_v,
        rigid_omega, p_i);
    TF inv_dt = 1 / dt;
    particle_dfsph_factor(p_i) *=
        inv_dt;  // TODO: move to compute_dfsph_factor?
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void divergence_solve_iteration(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_dfsph_factor, Variable<1, TF> particle_density_adv,
    Variable<1, TF> particle_kappa_v, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, Variable<2, TQ> particle_boundary,
    Variable<2, TQ> particle_boundary_kernel, Variable<2, TF3> particle_force,
    Variable<2, TF3> particle_torque, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, TF dt,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF b_i = particle_density_adv(p_i);
    TF k_i = b_i * particle_dfsph_factor(p_i);
    particle_kappa_v(p_i) += k_i;
    TF3 x_i = particle_x(p_i);
    TF3 dv{};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      const TF b_j = particle_density_adv(p_j);
      const TF kj = b_j * particle_dfsph_factor(p_j);

      const TF kSum = k_i + kj;
      if (fabs(kSum) > cn<TF>().dfsph_factor_epsilon) {
        const TF3 grad_p_j =
            -cn<TF>().particle_vol * displacement_cubic_kernel_grad(xixj);
        dv -= dt * kSum * grad_p_j;
      }
    }
    if (fabs(k_i) > cn<TF>().dfsph_factor_epsilon) {
      for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
        TQ boundary = particle_boundary(boundary_id, p_i);
        TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
        TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
        TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
        TF3 a = k_i * -grad_wvol;
        TF3 force = cn<TF>().particle_mass * a;
        dv -= dt * a;
        particle_force(boundary_id, p_i) += force;
        particle_torque(boundary_id, p_i) +=
            cross(bx_j - rigid_x(boundary_id), force);
      }
    }
    particle_v(p_i) += dv;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_divergence_solve_density_error(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density_adv, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, Variable<2, TQ> particle_boundary,
    Variable<2, TQ> particle_boundary_kernel, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, TF dt,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_density_adv(p_i) = compute_density_change<TQ, TF3, TF>(
        particle_x, particle_v, particle_neighbors, particle_num_neighbors,
        particle_boundary, particle_boundary_kernel, rigid_x, rigid_v,
        rigid_omega, p_i);
  });
}

template <typename TF>
__global__ void divergence_solve_finish(Variable<1, TF> particle_dfsph_factor,
                                        Variable<1, TF> particle_kappa_v, TF dt,
                                        U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_dfsph_factor(p_i) *= dt;
    particle_kappa_v(p_i) *= dt;
  });
}

template <typename TF3, typename TF>
__global__ void integrate_non_pressure_acceleration(Variable<1, TF3> particle_v,
                                                    Variable<1, TF3> particle_a,
                                                    TF dt, U num_particles) {
  forThreadMappedToElement(
      num_particles, [&](U p_i) { particle_v(p_i) += dt * particle_a(p_i); });
}

template <typename TQ, typename TF3, typename TF>
__global__ void warm_start_pressure_solve0(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<1, TF> particle_kappa,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    Variable<1, TF3> rigid_x, Variable<1, TF3> rigid_v,
    Variable<1, TF3> rigid_omega, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF density_adv = compute_density_adv(
        particle_x, particle_v, particle_density, particle_neighbors,
        particle_num_neighbors, particle_boundary, particle_boundary_kernel,
        rigid_x, rigid_v, rigid_omega, p_i, dt);
    particle_kappa(p_i) =
        (density_adv > 1.0)
            ? (static_cast<TF>(0.5) *
               max(particle_kappa(p_i), static_cast<TF>(-0.00025)) / (dt * dt))
            : static_cast<TF>(0);
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void warm_start_pressure_solve1(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_kappa, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, Variable<2, TQ> particle_boundary,
    Variable<2, TQ> particle_boundary_kernel, Variable<2, TF3> particle_force,
    Variable<2, TF3> particle_torque, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, TF dt,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    const TF3 x_i = particle_x(p_i);
    const TF k_i = particle_kappa(p_i);
    TF3 dv{};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      const TF k_sum = k_i + particle_kappa(p_j);
      if (fabs(k_sum) > cn<TF>().dfsph_factor_epsilon) {
        dv += dt * k_sum * cn<TF>().particle_vol *
              displacement_cubic_kernel_grad(xixj);
      }
    }
    if (fabs(k_i) > cn<TF>().dfsph_factor_epsilon) {
      for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
        TQ boundary = particle_boundary(boundary_id, p_i);
        TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
        TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
        TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
        TF3 a = k_i * -grad_wvol;
        TF3 force = cn<TF>().particle_mass * a;
        dv -= dt * a;
        particle_force(boundary_id, p_i) += force;
        particle_torque(boundary_id, p_i) +=
            cross(bx_j - rigid_x(boundary_id), force);
      }
    }
    particle_v(p_i) += dv;
  });
}

template <typename TQ, typename TF3, typename TF>
__device__ TF compute_density_adv(
    Variable<1, TF3>& particle_x, Variable<1, TF3>& particle_v,
    Variable<1, TF>& particle_density, Variable<2, TQ>& particle_neighbors,
    Variable<1, U>& particle_num_neighbors, Variable<2, TQ>& particle_boundary,
    Variable<2, TQ>& particle_boundary_kernel, Variable<1, TF3>& rigid_x,
    Variable<1, TF3>& rigid_v, Variable<1, TF3>& rigid_omega, U p_i, TF dt) {
  TF delta = 0;
  TF3 x_i = particle_x(p_i);
  TF3 v_i = particle_v(p_i);
  for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
       ++neighbor_id) {
    U p_j;
    TF3 xixj;
    extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
    delta += cn<TF>().particle_vol *
             dot(v_i - particle_v(p_j), displacement_cubic_kernel_grad(xixj));
  }

  for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
    TQ boundary = particle_boundary(boundary_id, p_i);
    TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
    TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
    TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
    TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
               rigid_v(boundary_id);
    delta += dot(v_i - velj, grad_wvol);
  }
  return max(particle_density(p_i) / cn<TF>().density0 + dt * delta,
             static_cast<TF>(1));
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_rho_adv(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<1, TF> particle_dfsph_factor,
    Variable<1, TF> particle_density_adv, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, Variable<2, TQ> particle_boundary,
    Variable<2, TQ> particle_boundary_kernel, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, TF dt,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_density_adv(p_i) = compute_density_adv(
        particle_x, particle_v, particle_density, particle_neighbors,
        particle_num_neighbors, particle_boundary, particle_boundary_kernel,
        rigid_x, rigid_v, rigid_omega, p_i, dt);
    particle_dfsph_factor(p_i) *= 1 / (dt * dt);
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void pressure_solve_iteration(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_dfsph_factor, Variable<1, TF> particle_density_adv,
    Variable<1, TF> particle_kappa, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, Variable<2, TQ> particle_boundary,
    Variable<2, TQ> particle_boundary_kernel, Variable<2, TF3> particle_force,
    Variable<2, TF3> particle_torque, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, TF dt,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    const TF b_i = particle_density_adv(p_i) - 1;
    const TF k_i = b_i * particle_dfsph_factor(p_i);
    particle_kappa(p_i) += k_i;
    const TF3 x_i = particle_x(p_i);
    TF3 dv{};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);

      const TF b_j = particle_density_adv(p_j) - 1;
      const TF k_sum = k_i + b_j * particle_dfsph_factor(p_j);
      if (fabs(k_sum) > cn<TF>().dfsph_factor_epsilon) {
        dv += dt * k_sum * cn<TF>().particle_vol *
              displacement_cubic_kernel_grad(xixj);
      }
    }
    if (fabs(k_i) > cn<TF>().dfsph_factor_epsilon) {
      for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
        TQ boundary = particle_boundary(boundary_id, p_i);
        TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
        TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
        TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
        TF3 a = k_i * -grad_wvol;
        TF3 force = cn<TF>().particle_mass * a;
        dv -= dt * a;
        particle_force(boundary_id, p_i) += force;
        particle_torque(boundary_id, p_i) +=
            cross(bx_j - rigid_x(boundary_id), force);
      }
    }
    particle_v(p_i) += dv;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_pressure_solve_density_error(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<1, TF> particle_density_adv,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    Variable<1, TF3> rigid_x, Variable<1, TF3> rigid_v,
    Variable<1, TF3> rigid_omega, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_density_adv(p_i) = compute_density_adv(
        particle_x, particle_v, particle_density, particle_neighbors,
        particle_num_neighbors, particle_boundary, particle_boundary_kernel,
        rigid_x, rigid_v, rigid_omega, p_i, dt);
  });
}

template <typename TF>
__global__ void pressure_solve_finish(Variable<1, TF> particle_kappa, TF dt,
                                      U num_particles) {
  forThreadMappedToElement(num_particles,
                           [&](U p_i) { particle_kappa(p_i) *= dt * dt; });
}

template <U wrap, typename TF3, typename TF>
__global__ void integrate_velocity(Variable<1, TF3> particle_x,
                                   Variable<1, TF3> particle_v, TF dt,
                                   U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    if constexpr (wrap == 0)
      particle_x(p_i) += dt * particle_v(p_i);
    else
      particle_x(p_i) = wrap_y(particle_x(p_i) + particle_v(p_i) * dt);
  });
}

template <typename TF3, typename TF>
__global__ void set_ethier_steinman(Variable<1, TF3> particle_x,
                                    Variable<1, TF3> particle_v, TF a, TF d,
                                    TF kinematic_viscosity, TF t,
                                    TF3 exclusion_min, TF3 exclusion_max,
                                    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    const TF3 x_i = particle_x(p_i);
    if (!within_box(x_i, exclusion_min, exclusion_max)) {
      TF exp_ax = exp(a * x_i.x);
      TF exp_ay = exp(a * x_i.y);
      TF exp_az = exp(a * x_i.z);
      TF exp_temporal = -a * exp(-d * d * kinematic_viscosity * t);
      TF ay_dz = a * x_i.y + d * x_i.z;
      TF ax_dy = a * x_i.x + d * x_i.y;
      TF az_dx = a * x_i.z + d * x_i.x;
      particle_v(p_i) =
          exp_temporal * TF3{exp_ax * sin(ay_dz) + exp_az * cos(ax_dy),
                             exp_ay * sin(az_dx) + exp_ax * cos(ay_dz),
                             exp_az * sin(ax_dy) + exp_ay * cos(az_dx)};
    }
  });
}

template <typename TF3>
__global__ void set_box_mask(Variable<1, TF3> particle_x, Variable<1, U> mask,
                             TF3 box_min, TF3 box_max, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    mask(p_i) = within_box(particle_x(p_i), box_min, box_max) ? 0 : 1;
  });
}

template <typename TF3>
__global__ void copy_kinematics_if_between(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF3> dest_x, Variable<1, TF3> dest_v, TF3 outer_box_min,
    TF3 outer_box_max, TF3 inner_box_min, TF3 inner_box_max, U num_particles,
    Variable<1, U> num_copied) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    const TF3 x_i = particle_x(p_i);
    if (within_box(x_i, outer_box_min, outer_box_max) &&
        !within_box(x_i, inner_box_min, inner_box_max)) {
      U dest_id = atomicAdd(&num_copied(0), 1);
      dest_x(dest_id) = x_i;
      dest_v(dest_id) = particle_v(p_i);
    }
  });
}

template <typename TF3>
__global__ void copy_kinematics_if_within(Variable<1, TF3> particle_x,
                                          Variable<1, TF3> particle_v,
                                          Variable<1, TF3> dest_x,
                                          Variable<1, TF3> dest_v, TF3 box_min,
                                          TF3 box_max, U num_particles,
                                          Variable<1, U> num_copied) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    const TF3 x_i = particle_x(p_i);
    if (within_box(x_i, box_min, box_max)) {
      U dest_id = atomicAdd(&num_copied(0), 1);
      dest_x(dest_id) = x_i;
      dest_v(dest_id) = particle_v(p_i);
    }
  });
}
template <typename TF3>
__global__ void copy_kinematics_if_within_masked(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, U> mask, U mask_keep_value, Variable<1, TF3> dest_x,
    Variable<1, TF3> dest_v, TF3 box_min, TF3 box_max, U num_particles,
    Variable<1, U> num_copied) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    const TF3 x_i = particle_x(p_i);
    const U mask_i = mask(p_i);
    if (within_box(x_i, box_min, box_max) && mask_i == mask_keep_value) {
      U dest_id = atomicAdd(&num_copied(0), 1);
      dest_x(dest_id) = x_i;
      dest_v(dest_id) = particle_v(p_i);
    }
  });
}

// rigid
template <typename TQ, typename TF3, typename TF, typename TDistance>
__global__ void collision_test(U i, U j, Variable<1, TF3> vertices_i,
                               Variable<1, U> num_contacts,
                               Variable<1, Contact<TF3, TF>> contacts,
                               TF mass_i, TF3 inertia_tensor_i, TF3 x_i,
                               TF3 v_i, TQ q_i, TF3 omega_i, TF mass_j,
                               TF3 inertia_tensor_j, TF3 x_j, TF3 v_j, TQ q_j,
                               TF3 omega_j, TF restitution, TF friction,

                               const TDistance distance, TF sign,

                               U num_vertices

) {
  forThreadMappedToElement(num_vertices, [&](U vertex_i) {
    Contact<TF3, TF> contact;
    TF3 vertex_local = vertices_i(vertex_i);

    contact.cp_i = rotate_using_quaternion(vertex_local, q_i) + x_i;
    TF3 vertex_local_wrt_j =
        rotate_using_quaternion(contact.cp_i - x_j, quaternion_conjugate(q_j));

    TF dist = distance.signed_distance(vertex_local_wrt_j) * sign -
              cn<TF>().contact_tolerance;
    TF3 n = distance.gradient(vertex_local_wrt_j, static_cast<TF>(1e-3)) * sign;
    n *= rsqrt(length_sqr(n));

    TF3 cp = vertex_local_wrt_j - dist * n;

    if (dist < 0) {
      U contact_insert_index = atomicAdd(&num_contacts(0), 1);
      if (contact_insert_index == cni.max_num_contacts) {
        printf("Reached the max. no. of contacts\n");
      }
      contact.i = i;
      contact.j = j;
      contact.cp_j = rotate_using_quaternion(cp, q_j) + x_j;
      contact.n = rotate_using_quaternion(n, q_j);

      contact.friction = friction;

      TF3 r_i = contact.cp_i - x_i;
      TF3 r_j = contact.cp_j - x_j;

      TF3 u_i = v_i + cross(omega_i, r_i);
      TF3 u_j = v_j + cross(omega_j, r_j);

      TF3 u_rel = u_i - u_j;
      TF u_rel_n = dot(contact.n, u_rel);

      contact.t = u_rel - u_rel_n * contact.n;
      TF tl2 = length_sqr(contact.t);
      if (tl2 > static_cast<TF>(1e-6)) {
        contact.t = normalize(contact.t);
      }

      calculate_congruent_matrix(1 / inertia_tensor_i, q_i,
                                 &(contact.iiwi_diag),
                                 &(contact.iiwi_off_diag));
      calculate_congruent_matrix(1 / inertia_tensor_j, q_j,
                                 &(contact.iiwj_diag),
                                 &(contact.iiwj_off_diag));

      TF3 k_i_diag, k_i_off_diag;
      TF3 k_j_diag, k_j_off_diag;
      calculate_congruent_k(contact.cp_i - x_i, mass_i, contact.iiwi_diag,
                            contact.iiwi_off_diag, &k_i_diag, &k_i_off_diag);
      calculate_congruent_k(contact.cp_j - x_j, mass_j, contact.iiwj_diag,
                            contact.iiwj_off_diag, &k_j_diag, &k_j_off_diag);
      TF3 k_diag = k_i_diag + k_j_diag;
      TF3 k_off_diag = k_i_off_diag + k_j_off_diag;

      contact.nkninv =
          1 / dot(contact.n, apply_congruent(contact.n, k_diag, k_off_diag));
      contact.pmax =
          1 / dot(contact.t, apply_congruent(contact.t, k_diag, k_off_diag)) *
          dot(u_rel, contact.t);  // note: error-prone when the magnitude of
                                  // tangent is small

      TF goal_u_rel_n = 0;
      if (u_rel_n < 0) {
        goal_u_rel_n = -restitution * u_rel_n;
      }

      contact.goalu = goal_u_rel_n;
      contact.impulse_sum = 0;
      contacts(contact_insert_index) = contact;
    }
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void collision_test(
    U i, U j, Variable<1, TF3> vertices_i, Variable<1, U> num_contacts,
    Variable<1, Contact<TF3, TF>> contacts, TF mass_i, TF3 inertia_tensor_i,
    TF3 x_i, TF3 v_i, TQ q_i, TF3 omega_i, TF mass_j, TF3 inertia_tensor_j,
    TF3 x_j, TF3 v_j, TQ q_j, TF3 omega_j, TF restitution, TF friction,

    Variable<1, TF> distance_nodes, TF3 domain_min, TF3 domain_max,
    U3 resolution, TF3 cell_size, U node_offset, TF sign,

    U num_vertices

) {
  forThreadMappedToElement(num_vertices, [&](U vertex_i) {
    Contact<TF3, TF> contact;
    TF3 vertex_local = vertices_i(vertex_i);

    contact.cp_i = rotate_using_quaternion(vertex_local, q_i) + x_i;
    TF3 vertex_local_wrt_j =
        rotate_using_quaternion(contact.cp_i - x_j, quaternion_conjugate(q_j));

    TF3 n;
    TF dist = collision_find_dist_normal(
        &distance_nodes, domain_min, domain_max, resolution, cell_size, 0, sign,
        cn<TF>().contact_tolerance, vertex_local_wrt_j, &n);
    TF3 cp = vertex_local_wrt_j - dist * n;

    if (dist < 0) {
      U contact_insert_index = atomicAdd(&num_contacts(0), 1);
      if (contact_insert_index == cni.max_num_contacts) {
        printf("Reached the max. no. of contacts\n");
      }
      contact.i = i;
      contact.j = j;
      contact.cp_j = rotate_using_quaternion(cp, q_j) + x_j;
      contact.n = rotate_using_quaternion(n, q_j);

      contact.friction = friction;

      TF3 r_i = contact.cp_i - x_i;
      TF3 r_j = contact.cp_j - x_j;

      TF3 u_i = v_i + cross(omega_i, r_i);
      TF3 u_j = v_j + cross(omega_j, r_j);

      TF3 u_rel = u_i - u_j;
      TF u_rel_n = dot(contact.n, u_rel);

      contact.t = u_rel - u_rel_n * contact.n;
      TF tl2 = length_sqr(contact.t);
      if (tl2 > static_cast<TF>(1e-6)) {
        contact.t = normalize(contact.t);
      }

      calculate_congruent_matrix(1 / inertia_tensor_i, q_i,
                                 &(contact.iiwi_diag),
                                 &(contact.iiwi_off_diag));
      calculate_congruent_matrix(1 / inertia_tensor_j, q_j,
                                 &(contact.iiwj_diag),
                                 &(contact.iiwj_off_diag));

      TF3 k_i_diag, k_i_off_diag;
      TF3 k_j_diag, k_j_off_diag;
      calculate_congruent_k(contact.cp_i - x_i, mass_i, contact.iiwi_diag,
                            contact.iiwi_off_diag, &k_i_diag, &k_i_off_diag);
      calculate_congruent_k(contact.cp_j - x_j, mass_j, contact.iiwj_diag,
                            contact.iiwj_off_diag, &k_j_diag, &k_j_off_diag);
      TF3 k_diag = k_i_diag + k_j_diag;
      TF3 k_off_diag = k_i_off_diag + k_j_off_diag;

      contact.nkninv =
          1 / dot(contact.n, apply_congruent(contact.n, k_diag, k_off_diag));
      contact.pmax =
          1 / dot(contact.t, apply_congruent(contact.t, k_diag, k_off_diag)) *
          dot(u_rel, contact.t);  // note: error-prone when the magnitude of
                                  // tangent is small

      TF goal_u_rel_n = 0;
      if (u_rel_n < 0) {
        goal_u_rel_n = -restitution * u_rel_n;
      }

      contact.goalu = goal_u_rel_n;
      contact.impulse_sum = 0;
      contacts(contact_insert_index) = contact;
    }
  });
}

// fluid control
template <typename TF3, typename TF>
__global__ void drive_linear(Variable<1, TF3> particle_x,
                             Variable<1, TF3> particle_v,
                             Variable<1, TF3> particle_a,
                             Variable<1, TF3> usher_x, Variable<1, TF3> usher_v,
                             Variable<1, TF> drive_kernel_radius,
                             Variable<1, TF> drive_strength, U num_ushers,
                             U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    const TF3 x_i = particle_x(p_i);
    const TF3 v_i = particle_v(p_i);
    TF3 da{0};
    for (U usher_id = 0; usher_id < num_ushers; ++usher_id) {
      TF3 c_x = usher_x(usher_id);
      TF3 c_v = usher_v(usher_id);
      TF c_kernel = drive_kernel_radius(usher_id);
      TF c_strength = drive_strength(usher_id);
      da += c_strength * c_kernel * c_kernel * c_kernel * 8 * 0.1 *
            displacement_cubic_kernel(x_i - c_x, c_kernel) * (c_v - v_i);
    }
    particle_a(p_i) += da;
  });
}

// statistics
template <typename TF>
__global__ void compute_inverse(Variable<1, TF> source, Variable<1, TF> dest,
                                U n) {
  forThreadMappedToElement(n, [&](U i) { dest(i) = 1 / source(i); });
}

template <typename TF3, typename TF>
__global__ void compute_magnitude(Variable<1, TF3> v, Variable<1, TF> magnitude,
                                  U n) {
  forThreadMappedToElement(n, [&](U i) { magnitude(i) = length(v(i)); });
}

template <typename TF3>
__global__ void count_out_of_grid(Variable<1, TF3> particle_x,
                                  Variable<1, U> out_of_grid_count,
                                  U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    I3 ipos = get_ipos(particle_x(p_i));
    U pid_insert_index;
    if (!within_grid(ipos)) {
      atomicAdd(&out_of_grid_count(0), 1);
    }
  });
}

template <typename TQ, typename TF3, typename TF, typename TQuantity>
__global__ void sample_fluid(
    Variable<1, TF3> sample_x, Variable<1, TF3> particle_x,
    Variable<1, TF> particle_density, Variable<1, TQuantity> particle_quantity,
    Variable<2, TQ> sample_neighbors, Variable<1, U> sample_num_neighbors,
    Variable<1, TQuantity> sample_quantity, U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TQuantity result{0};
    TF3 x_i = sample_x(p_i);
    for (U neighbor_id = 0; neighbor_id < sample_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(sample_neighbors(p_i, neighbor_id), xixj, p_j);
      result += cn<TF>().particle_mass / particle_density(p_j) *
                displacement_cubic_kernel(xixj) * particle_quantity(p_j);
    }
    sample_quantity(p_i) = result;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void sample_velocity(
    Variable<1, TF3> sample_x, Variable<1, TF3> particle_x,
    Variable<1, TF> particle_density, Variable<1, TF3> particle_v,
    Variable<2, TQ> sample_neighbors, Variable<1, U> sample_num_neighbors,
    Variable<1, TF3> sample_v, Variable<2, TQ> sample_boundary,
    Variable<2, TQ> sample_boundary_kernel, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF3 result{0};
    TF3 x_i = sample_x(p_i);
    for (U neighbor_id = 0; neighbor_id < sample_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(sample_neighbors(p_i, neighbor_id), xixj, p_j);
      result += cn<TF>().particle_mass / particle_density(p_j) *
                displacement_cubic_kernel(xixj) * particle_v(p_j);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = sample_boundary(boundary_id, p_i);
      TQ boundary_kernel = sample_boundary_kernel(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);

      TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
                 rigid_v(boundary_id);
      result += velj * boundary_kernel.w;
    }
    sample_v(p_i) = result;
  });
}

// Graphical post-processing
template <typename TFDest, typename TF3Dest, typename TF3Source>
__global__ void convert_fp3(Variable<1, TF3Dest> dest,
                            Variable<1, TF3Source> source, U n) {
  forThreadMappedToElement(n, [&](U i) {
    TF3Source v = source(i);
    dest(i) = TF3Dest{static_cast<TFDest>(v.x), static_cast<TFDest>(v.y),
                      static_cast<TFDest>(v.z)};
  });
}

template <typename TF3, typename TF>
__global__ void normalize_vector_magnitude(Variable<1, TF3> v,
                                           Variable<1, TF> normalized,
                                           TF lower_bound, TF upper_bound,
                                           U n) {
  forThreadMappedToElement(n, [&](U i) {
    normalized(i) = (length(v(i)) - lower_bound) / (upper_bound - lower_bound);
  });
}

template <typename TF>
__global__ void scale(Variable<1, TF> original, Variable<1, TF> scaled,
                      TF lower_bound, TF upper_bound, U n) {
  forThreadMappedToElement(n, [&](U i) {
    scaled(i) = (original(i) - lower_bound) / (upper_bound - lower_bound);
  });
}

template <typename TF>
class Runner {
 public:
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  Runner() : default_block_size_(256) {
    Allocator::abort_if_error(cudaEventCreate(&abs_start_));
    Allocator::abort_if_error(cudaEventRecord(abs_start_));
    if (const char* default_block_size_str =
            std::getenv("AL_DEFAULT_BLOCK_SIZE")) {
      default_block_size_ = std::stoul(default_block_size_str);
    }
    load_optimal_block_size();
  }
  virtual ~Runner() {
    summarize();
    std::stringstream filename_stream;
    filename_stream << ".alcache/";
    if (optimal_block_size_dict_.empty()) {
      filename_stream << default_block_size_;
    } else {
      filename_stream << "optimized";
    }
    filename_stream << ".yaml";
    save_stat(filename_stream.str().c_str());
    cudaEventDestroy(abs_start_);
  }
  template <typename T, typename BinaryFunction>
  static T reduce(void* ptr, U num_elements, T init, BinaryFunction binary_op,
                  U offset = 0) {
    return thrust::reduce(
        thrust::device_ptr<T>(static_cast<T*>(ptr)) + offset,
        thrust::device_ptr<T>(static_cast<T*>(ptr)) + (offset + num_elements),
        init, binary_op);
  }

  template <typename T>
  static T sum(void* ptr, U num_elements, U offset = 0) {
    return reduce(ptr, num_elements, T{}, thrust::plus<T>(), offset);
  }

  template <U D, typename M>
  static M sum(Variable<D, M> var, U num_elements, U offset = 0) {
    return reduce(var.ptr_, num_elements, M{}, thrust::plus<M>(), offset);
  }

  template <U D, typename M>
  static M max(Variable<D, M> var, U num_elements, U offset = 0) {
    return reduce(var.ptr_, num_elements, std::numeric_limits<M>::lowest(),
                  thrust::maximum<M>(), offset);
  }

  template <U D, typename M>
  static M min(Variable<D, M> var, U num_elements, U offset = 0) {
    return reduce(var.ptr_, num_elements, std::numeric_limits<M>::max(),
                  thrust::minimum<M>(), offset);
  }

  template <class Lambda>
  static void launch(U n, U desired_block_size, Lambda f) {
    if (n == 0) return;
    U block_size = std::min(n, desired_block_size);
    U grid_size = (n + block_size - 1) / block_size;
    f(grid_size, block_size);
    Allocator::abort_if_error(cudaGetLastError());
  }

  template <class Lambda, typename Func>
  static void launch_occupancy(U n, Lambda f, Func func) {
    if (n == 0) return;
    int min_grid_size, block_size;
    Allocator::abort_if_error(
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func));
    launch(n, block_size, f);
  }

  template <class Lambda>
  void launch(U n, U desired_block_size, Lambda f, std::string name) {
    if (n == 0) return;
    cudaEvent_t event_start, event_stop;
    float ellapsed;
    float started;
    Allocator::abort_if_error(cudaEventCreate(&event_start));
    Allocator::abort_if_error(cudaEventCreate(&event_stop));
    U block_size = std::min(n, desired_block_size);
    U grid_size = (n + block_size - 1) / block_size;
    Allocator::abort_if_error(cudaGetLastError());
    Allocator::abort_if_error(cudaEventRecord(event_start));
    f(grid_size, block_size);
    Allocator::abort_if_error(cudaGetLastError());
    Allocator::abort_if_error(cudaEventRecord(event_stop));
    Allocator::abort_if_error(cudaEventSynchronize(event_stop));
    Allocator::abort_if_error(
        cudaEventElapsedTime(&started, abs_start_, event_start));
    Allocator::abort_if_error(
        cudaEventElapsedTime(&ellapsed, event_start, event_stop));
    launch_dict_[name][desired_block_size / kWarpSize - 1].emplace_back(
        started, ellapsed);
    Allocator::abort_if_error(cudaEventDestroy(event_start));
    Allocator::abort_if_error(cudaEventDestroy(event_stop));
  }

  template <class Lambda, typename Func>
  void launch(U n, Lambda f, std::string name, Func func) {
    if (n == 0) return;
    cudaFuncAttributes attr;
    Allocator::abort_if_error(cudaFuncGetAttributes(&attr, func));
    int block_size = default_block_size_;
    if (!optimal_block_size_dict_.empty()) {
      if (optimal_block_size_dict_.find(name) ==
          optimal_block_size_dict_.end()) {
        std::cerr << "Optimal block size for " << name << " not found"
                  << std::endl;
        abort();
      }
      block_size = optimal_block_size_dict_[name];
    }
    if (block_size >
        attr.maxThreadsPerBlock) {  // max. block size can be smaller than 1024
                                    // depending on the kernel
      block_size = attr.maxThreadsPerBlock;
    }
    launch(n, block_size, f, name);
  }

  void summarize() {
    for (std::pair<std::string, std::array<std::vector<LaunchRecord>,
                                           kNumBlockSizeCandidates>> const&
             item : launch_dict_) {
      for (U i = 0; i < kNumBlockSizeCandidates; ++i) {
        std::vector<LaunchRecord> const& launch_records = item.second[i];
        launch_stat_dict_[item.first][i] =
            launch_records.empty()
                ? -1
                : std::accumulate(launch_records.begin(), launch_records.end(),
                                  0.f,
                                  [](float acc, LaunchRecord const& record) {
                                    return acc + record.second;
                                  }) /
                      launch_records.size();
      }
    }
  }
  void save_stat(const char* filename) const {
    std::ofstream stream(filename, std::ios::trunc);
    if (!stream) {
      std::cerr << "Failed writing Runner statistics to " << filename
                << std::endl;
      return;
    }
    for (auto const& item : launch_stat_dict_) {
      stream << item.first << ": ";
      stream << "[";
      for (auto const& mean : item.second) {
        stream << std::setprecision(std::numeric_limits<float>::max_digits10)
               << mean << ", ";
      }
      stream << "]" << std::endl;
    }
  };
  void load_optimal_block_size() {
    std::ifstream stream(".alcache/optimal.yaml");
    if (!stream.is_open()) return;
    std::string line;
    std::stringstream line_stream;
    std::string function_name;
    std::string elapsed_str;
    stream.exceptions(std::ios_base::badbit);
    while (std::getline(stream, line)) {
      line_stream.clear();
      line_stream.str(line);
      std::getline(line_stream, function_name, ':');
      std::getline(line_stream, elapsed_str);
      optimal_block_size_dict_[function_name] = std::stoul(elapsed_str.c_str());
    }
  };
  void launch_create_fluid_block(Variable<1, TF3>& particle_x, U num_particles,
                                 U offset, int mode, TF3 box_min, TF3 box_max) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          create_fluid_block<TF3, TF><<<grid_size, block_size>>>(
              particle_x, num_particles, offset, mode, box_min, box_max);
        },
        "create_fluid_block", create_fluid_block<TF3, TF>);
  }
  static U get_fluid_block_num_particles(int mode, TF3 box_min, TF3 box_max,
                                         TF particle_radius) {
    I3 steps;
    TF diameter;
    TF3 diff;
    TF xshift, yshift;
    get_fluid_block_attr(mode, box_min, box_max, particle_radius, steps,
                         diameter, diff, xshift, yshift);
    return steps.x * steps.y * steps.z;
  }
  static U get_fluid_cylinder_num_particles(TF radius, TF y_min, TF y_max,
                                            TF particle_radius) {
    U sqrt_n;
    U n;
    U steps_y;
    TF diameter;
    get_fluid_cylinder_attr(radius, y_min, y_max, particle_radius, sqrt_n, n,
                            steps_y, diameter);
    return n * steps_y;
  }
  void launch_create_fluid_cylinder_sunflower(Variable<1, TF3>& particle_x,
                                              U num_particles, TF radius,
                                              U num_particles_per_slice,
                                              TF slice_distance, TF y_min) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          create_fluid_cylinder_sunflower<TF3, TF><<<grid_size, block_size>>>(
              particle_x, num_particles, radius, num_particles_per_slice,
              slice_distance, y_min);
        },
        "create_fluid_cylinder_sunflower",
        create_fluid_cylinder_sunflower<TF3, TF>);
  }

  void launch_create_fluid_cylinder(Variable<1, TF3>& particle_x,
                                    U num_particles, U offset, TF radius,
                                    TF y_min, TF y_max) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          create_fluid_cylinder<TF3, TF><<<grid_size, block_size>>>(
              particle_x, num_particles, offset, radius, y_min, y_max);
        },
        "create_fluid_cylinder", create_fluid_cylinder<TF3, TF>);
  }
  void launch_compute_particle_boundary(
      dg::Distance<TF3, TF> const& virtual_dist,
      Variable<1, TF> const& volume_nodes,
      Variable<1, TF> const& distance_nodes, TF3 const& rigid_x,
      TQ const& rigid_q, U boundary_id, TF3 const& domain_min,
      TF3 const& domain_max, U3 const& resolution, TF3 const& cell_size,
      U num_nodes, U node_offset, TF sign, TF thickness, TF dt,
      Variable<1, TF3>& particle_x, Variable<2, TQ>& particle_boundary,
      Variable<2, TQ>& particle_boundary_kernel, U num_particles) {
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    if (TBoxDistance const* distance =
            dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<0><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, num_nodes, 0,
                sign, thickness, dt, particle_x, particle_boundary,
                particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(BoxDistance)",
          compute_particle_boundary_analytic<0, TQ, TF3, TF, TBoxDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<0><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, num_nodes, 0,
                sign, thickness, dt, particle_x, particle_boundary,
                particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(SphereDistance)",
          compute_particle_boundary_analytic<0, TQ, TF3, TF, TSphereDistance>);
    } else if (TInfiniteCylinderDistance const* distance =
                   dynamic_cast<TInfiniteCylinderDistance const*>(
                       &virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<1><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, num_nodes, 0,
                sign, thickness, dt, particle_x, particle_boundary,
                particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(InfiniteCylinderDistance)",
          compute_particle_boundary_analytic<1, TQ, TF3, TF,
                                             TInfiniteCylinderDistance>);
    } else {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary<<<grid_size, block_size>>>(
                volume_nodes, distance_nodes, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, num_nodes, 0,
                sign, thickness, dt, particle_x, particle_boundary,
                particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary", compute_particle_boundary<TQ, TF3, TF>);
    }
  }
  void launch_update_particle_grid(Variable<1, TF3>& particle_x,
                                   Variable<4, TQ>& pid,
                                   Variable<3, U>& pid_length,
                                   U num_particles) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          update_particle_grid<<<grid_size, block_size>>>(
              particle_x, pid, pid_length, num_particles);
        },
        "update_particle_grid", update_particle_grid<TQ, TF3>);
  }
  template <U wrap>
  void launch_make_neighbor_list(Variable<1, TF3>& sample_x,
                                 Variable<4, TQ>& pid,
                                 Variable<3, U>& pid_length,
                                 Variable<2, TQ>& sample_neighbors,
                                 Variable<1, U>& sample_num_neighbors,
                                 U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          make_neighbor_list<wrap><<<grid_size, block_size>>>(
              sample_x, pid, pid_length, sample_neighbors, sample_num_neighbors,
              num_samples);
        },
        "make_neighbor_list", make_neighbor_list<wrap, TQ, TF3>);
  }
  void launch_compute_density(Variable<1, TF3>& particle_x,
                              Variable<2, TQ>& particle_neighbors,
                              Variable<1, U>& particle_num_neighbors,
                              Variable<1, TF>& particle_density,
                              Variable<2, TQ>& particle_boundary_kernel,
                              U num_particles) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_density<<<grid_size, block_size>>>(
              particle_x, particle_neighbors, particle_num_neighbors,
              particle_density, particle_boundary_kernel, num_particles);
        },
        "compute_density", compute_density<TQ, TF3, TF>);
  }
  template <typename TQuantity>
  void launch_sample_fluid(Variable<1, TF3>& sample_x,
                           Variable<1, TF3>& particle_x,
                           Variable<1, TF>& particle_density,
                           Variable<1, TQuantity>& particle_quantity,
                           Variable<2, TQ>& sample_neighbors,
                           Variable<1, U>& sample_num_neighbors,
                           Variable<1, TQuantity>& sample_quantity,
                           U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          sample_fluid<<<grid_size, block_size>>>(
              sample_x, particle_x, particle_density, particle_quantity,
              sample_neighbors, sample_num_neighbors, sample_quantity,
              num_samples);
        },
        "sample_fluid", sample_fluid<TQ, TF3, TF, TQuantity>);
  }
  void launch_sample_velocity(
      Variable<1, TF3>& sample_x, Variable<1, TF3>& particle_x,
      Variable<1, TF>& particle_density, Variable<1, TF3>& particle_v,
      Variable<2, TQ>& sample_neighbors, Variable<1, U>& sample_num_neighbors,
      Variable<1, TF3>& sample_v, Variable<2, TQ>& sample_boundary,
      Variable<2, TQ>& sample_boundary_kernel, Variable<1, TF3> rigid_x,
      Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          sample_velocity<<<grid_size, block_size>>>(
              sample_x, particle_x, particle_density, particle_v,
              sample_neighbors, sample_num_neighbors, sample_v, sample_boundary,
              sample_boundary_kernel, rigid_x, rigid_v, rigid_omega,
              num_samples);
        },
        "sample_velocity", sample_velocity<TQ, TF3, TF>);
  }
  void launch_sample_density(Variable<1, TF3>& sample_x,
                             Variable<2, TQ>& sample_neighbors,
                             Variable<1, U>& sample_num_neighbors,
                             Variable<1, TF>& sample_density,
                             Variable<2, TQ>& sample_boundary_kernel,
                             U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          sample_position_density<<<grid_size, block_size>>>(
              sample_x, sample_neighbors, sample_num_neighbors, sample_density,
              sample_boundary_kernel, num_samples);
        },
        "sample_position_density", sample_position_density<TQ, TF3, TF>);
  }
  void launch_copy_kinematics_if_within(
      Variable<1, TF3>& particle_x, Variable<1, TF3>& particle_v,
      Variable<1, TF3>& dest_x, Variable<1, TF3>& dest_v, TF3 const& box_min,
      TF3 const& box_max, U num_particles, Variable<1, U>& num_copied) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          copy_kinematics_if_within<<<grid_size, block_size>>>(
              particle_x, particle_v, dest_x, dest_v, box_min, box_max,
              num_particles, num_copied);
        },
        "copy_kinematics_if_within", copy_kinematics_if_within<TF3>);
  }
  void launch_copy_kinematics_if_within_masked(
      Variable<1, TF3>& particle_x, Variable<1, TF3>& particle_v,
      Variable<1, U>& mask, U mask_keep_value, Variable<1, TF3>& dest_x,
      Variable<1, TF3>& dest_v, TF3 const& box_min, TF3 const& box_max,
      U num_particles, Variable<1, U>& num_copied) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          copy_kinematics_if_within_masked<<<grid_size, block_size>>>(
              particle_x, particle_v, mask, mask_keep_value, dest_x, dest_v,
              box_min, box_max, num_particles, num_copied);
        },
        "copy_kinematics_if_within_masked",
        copy_kinematics_if_within_masked<TF3>);
  }
  void launch_copy_kinematics_if_between(
      Variable<1, TF3>& particle_x, Variable<1, TF3>& particle_v,
      Variable<1, TF3>& dest_x, Variable<1, TF3>& dest_v,
      TF3 const& outer_box_min, TF3 const& outer_box_max,
      TF3 const& inner_box_min, TF3 const& inner_box_max, U num_particles,
      Variable<1, U>& num_copied) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          copy_kinematics_if_between<<<grid_size, block_size>>>(
              particle_x, particle_v, dest_x, dest_v, outer_box_min,
              outer_box_max, inner_box_min, inner_box_max, num_particles,
              num_copied);
        },
        "copy_kinematics_if_between", copy_kinematics_if_between<TF3>);
  }
  void launch_collision_test(
      dg::Distance<TF3, TF> const& virtual_dist,
      Variable<1, TF> const& distance_nodes, U i, U j,
      Variable<1, TF3>& vertices_i, Variable<1, U>& num_contacts,
      Variable<1, Contact<TF3, TF>>& contacts, TF const& mass_i,
      TF3 const& inertia_tensor_i, TF3 const& x_i, TF3 const& v_i,
      TQ const& q_i, TF3 const& omega_i, TF mass_j, TF3 const& inertia_tensor_j,
      TF3 const& x_j, TF3 const& v_j, TQ const& q_j, TF3 const& omega_j,
      TF restitution, TF friction,

      TF3 const& domain_min, TF3 const& domain_max, U3 const& resolution,
      TF3 const& cell_size, U node_offset, TF sign,

      U num_vertices_i) {
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    if (TBoxDistance const* distance =
            dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_vertices_i,
          [&](U grid_size, U block_size) {
            collision_test<<<grid_size, block_size>>>(
                i, j, vertices_i, num_contacts, contacts, mass_i,
                inertia_tensor_i, x_i, v_i, q_i, omega_i, mass_j,
                inertia_tensor_j, x_j, v_j, q_j, omega_j, restitution, friction,
                *distance, sign, num_vertices_i);
          },
          "collision_test(BoxDistance)",
          collision_test<TQ, TF3, TF, TBoxDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_vertices_i,
          [&](U grid_size, U block_size) {
            collision_test<<<grid_size, block_size>>>(
                i, j, vertices_i, num_contacts, contacts, mass_i,
                inertia_tensor_i, x_i, v_i, q_i, omega_i, mass_j,
                inertia_tensor_j, x_j, v_j, q_j, omega_j, restitution, friction,
                *distance, sign, num_vertices_i);
          },
          "collision_test(SphereDistance)",
          collision_test<TQ, TF3, TF, TSphereDistance>);
    } else if (TInfiniteCylinderDistance const* distance =
                   dynamic_cast<TInfiniteCylinderDistance const*>(
                       &virtual_dist)) {
      launch(
          num_vertices_i,
          [&](U grid_size, U block_size) {
            collision_test<<<grid_size, block_size>>>(
                i, j, vertices_i, num_contacts, contacts, mass_i,
                inertia_tensor_i, x_i, v_i, q_i, omega_i, mass_j,
                inertia_tensor_j, x_j, v_j, q_j, omega_j, restitution, friction,
                *distance, sign, num_vertices_i);
          },
          "collision_test(InfiniteCylinderDistance)",
          collision_test<TQ, TF3, TF, TInfiniteCylinderDistance>);
    } else {
      launch(
          num_vertices_i,
          [&](U grid_size, U block_size) {
            collision_test<<<grid_size, block_size>>>(
                i, j, vertices_i, num_contacts, contacts, mass_i,
                inertia_tensor_i, x_i, v_i, q_i, omega_i, mass_j,
                inertia_tensor_j, x_j, v_j, q_j, omega_j, restitution, friction,
                distance_nodes, domain_min, domain_max, resolution, cell_size,
                node_offset, sign, num_vertices_i);
          },
          "collision_test", collision_test<TQ, TF3, TF>);
    }
  }

  using LaunchRecord = std::pair<float, float>;
  constexpr static U kWarpSize = 32;
  constexpr static U kMaxBlockSize = 1024;  // compute capability >= 2
  constexpr static U kNumBlockSizeCandidates = kMaxBlockSize / kWarpSize;
  U default_block_size_;
  std::unordered_map<std::string, std::array<std::vector<LaunchRecord>,
                                             kNumBlockSizeCandidates>>
      launch_dict_;
  std::unordered_map<std::string, std::array<float, kNumBlockSizeCandidates>>
      launch_stat_dict_;
  std::unordered_map<std::string, U> optimal_block_size_dict_;
  cudaEvent_t abs_start_;
};

}  // namespace alluvion

#endif /* ALLUVION_RUNNER_HPP */
