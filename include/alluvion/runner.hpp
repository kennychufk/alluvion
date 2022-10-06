#ifndef ALLUVION_RUNNER_HPP
#define ALLUVION_RUNNER_HPP
#include <cooperative_groups.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "alluvion/constants.hpp"
#include "alluvion/contact.hpp"
#include "alluvion/data_type.hpp"
#include "alluvion/dg/box_distance.hpp"
#include "alluvion/dg/box_shell_distance.hpp"
#include "alluvion/dg/capsule_distance.hpp"
#include "alluvion/dg/cylinder_distance.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/infinite_tube_distance.hpp"
#include "alluvion/dg/mesh_distance.hpp"
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

template <typename TQ, typename TF3>
constexpr __device__ __host__ void quaternion_to_matrix(TQ const& q, TF3& m0,
                                                        TF3& m1, TF3& m2) {
  m0.x = 1 - 2 * (fma(q.y, q.y, q.z * q.z));
  m0.y = 2 * fma(q.x, q.y, -q.w * q.z);
  m0.z = 2 * fma(q.x, q.z, q.w * q.y);
  m1.x = 2 * fma(q.x, q.y, +q.w * q.z);
  m1.y = 1 - 2 * fma(q.x, q.x, q.z * q.z);
  m1.z = 2 * fma(q.y, q.z, -q.w * q.x);
  m2.x = 2 * fma(q.x, q.z, -q.w * q.y);
  m2.y = 2 * fma(q.y, q.z, q.w * q.x);
  m2.z = 1 - 2 * fma(q.x, q.x, q.y * q.y);
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
constexpr __device__ TF3 symmetric_matrix_product_diag(TF3 const& diag,
                                                       TF3 const& off_diag) {
  return TF3{
      diag.x * diag.x + off_diag.x * off_diag.x + off_diag.y * off_diag.y,
      off_diag.x * off_diag.x + diag.y * diag.y + off_diag.z * off_diag.z,
      off_diag.y * off_diag.y + off_diag.z * off_diag.z + diag.z * diag.z};
}

template <typename TF3>
constexpr __device__ TF3
symmetric_matrix_product_off_diag(TF3 const& diag, TF3 const& off_diag) {
  return TF3{
      diag.x * off_diag.x + diag.y * off_diag.x + off_diag.y * off_diag.z,
      diag.x * off_diag.y + off_diag.x * off_diag.z + diag.z * off_diag.y,
      off_diag.x * off_diag.y + diag.y * off_diag.z + off_diag.z * diag.z};
}

template <typename TF3>
constexpr __device__ void matrix_multiply(TF3 const& a0, TF3 const& a1,
                                          TF3 const& a2, TF3 const& b0,
                                          TF3 const& b1, TF3 const& b2, TF3& c0,
                                          TF3& c1, TF3& c2) {
  c0.x = fma(a0.x, b0.x, fma(a0.y, b1.x, a0.z * b2.x));
  c0.y = fma(a0.x, b0.y, fma(a0.y, b1.y, a0.z * b2.y));
  c0.z = fma(a0.x, b0.z, fma(a0.y, b1.z, a0.z * b2.z));
  c1.x = fma(a1.x, b0.x, fma(a1.y, b1.x, a1.z * b2.x));
  c1.y = fma(a1.x, b0.y, fma(a1.y, b1.y, a1.z * b2.y));
  c1.z = fma(a1.x, b0.z, fma(a1.y, b1.z, a1.z * b2.z));
  c2.x = fma(a2.x, b0.x, fma(a2.y, b1.x, a2.z * b2.x));
  c2.y = fma(a2.x, b0.y, fma(a2.y, b1.y, a2.z * b2.y));
  c2.z = fma(a2.x, b0.z, fma(a2.y, b1.z, a2.z * b2.z));
}

template <typename TF3>
constexpr __device__ void matrix_multiply(TF3 const& diag, TF3 const& off_diag,
                                          TF3 const& b0, TF3 const& b1,
                                          TF3 const& b2, TF3& c0, TF3& c1,
                                          TF3& c2) {
  c0.x = fma(diag.x, b0.x, fma(off_diag.x, b1.x, off_diag.y * b2.x));
  c0.y = fma(diag.x, b0.y, fma(off_diag.x, b1.y, off_diag.y * b2.y));
  c0.z = fma(diag.x, b0.z, fma(off_diag.x, b1.z, off_diag.y * b2.z));
  c1.x = fma(off_diag.x, b0.x, fma(diag.y, b1.x, off_diag.z * b2.x));
  c1.y = fma(off_diag.x, b0.y, fma(diag.y, b1.y, off_diag.z * b2.y));
  c1.z = fma(off_diag.x, b0.z, fma(diag.y, b1.z, off_diag.z * b2.z));
  c2.x = fma(off_diag.y, b0.x, fma(off_diag.z, b1.x, diag.z * b2.x));
  c2.y = fma(off_diag.y, b0.y, fma(off_diag.z, b1.y, diag.z * b2.y));
  c2.z = fma(off_diag.y, b0.z, fma(off_diag.z, b1.z, diag.z * b2.z));
}

template <typename TF3, typename TF2>
constexpr __device__ TF2 approximate_givens(TF3 const& diag,
                                            TF3 const& off_diag) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  constexpr TF kGivensGamma =
      TF(5.82842712474619009760337744842e+00L);                     // sqrt(8)+3
  constexpr TF kCStar = TF(0.923879532511286756128183189397e+00L);  // cos(pi/8)
  constexpr TF kSStar = TF(0.382683432365089771728459984030e+00L);  // sin(pi/8)

  TF2 g{2 * (diag.x - diag.y), off_diag.x};
  bool b = kGivensGamma * g.y * g.y < g.x * g.x;
  TF w = rsqrt(fma(g.x, g.x, g.y * g.y));
  if (w != w) b = false;
  return TF2{b ? w * g.x : kCStar, b ? w * g.y : kSStar};
}

template <typename TQ, typename TF3>
constexpr __device__ void jacobi_conjugation(const int x, const int y,
                                             const int z, TF3& diag,
                                             TF3& off_diag, TQ& q) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float2, double2> TF2;
  TF2 g = approximate_givens<TF3, TF2>(diag, off_diag);
  TF scale = 1 / fma(g.x, g.x, g.y * g.y);
  TF a = fma(g.x, g.x, -g.y * g.y) * scale;
  TF b = 2 * g.y * g.x * scale;

  TF3 _diag = diag;
  TF3 _off_diag = off_diag;
  diag.x = fma(a, fma(a, _diag.x, b * _off_diag.x),
               b * (fma(a, _off_diag.x, b * _diag.y)));
  off_diag.x = fma(a, fma(-b, _diag.x, a * _off_diag.x),
                   b * (fma(-b, _off_diag.x, a * _diag.y)));
  diag.y = fma(-b, fma(-b, _diag.x, a * _off_diag.x),
               a * (fma(-b, _off_diag.x, a * _diag.y)));
  off_diag.y = fma(a, _off_diag.y, b * _off_diag.z);
  off_diag.z = fma(-b, _off_diag.y, a * _off_diag.z);
  diag.z = _diag.z;
  // update cumulative rotation qV
  TF tmp[3];
  tmp[0] = q.x * g.y;
  tmp[1] = q.y * g.y;
  tmp[2] = q.z * g.y;
  g.y *= q.w;
  // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1)) for (p,q) =
  // ((0,1),(1,2),(0,2))
  (reinterpret_cast<TF*>(&q))[z] =
      fma((reinterpret_cast<TF const*>(&q))[z], g.x, g.y);
  q.w = fma(q.w, g.x, -tmp[z]);  // w
  (reinterpret_cast<TF*>(&q))[x] =
      fma((reinterpret_cast<TF const*>(&q))[x], g.x, tmp[y]);
  (reinterpret_cast<TF*>(&q))[y] =
      fma((reinterpret_cast<TF const*>(&q))[y], g.x, -tmp[x]);
  // re-arrange matrix for next iteration
  _diag.x = diag.y;
  _off_diag.x = off_diag.z;
  _diag.y = diag.z;
  _off_diag.y = off_diag.x;
  _off_diag.z = off_diag.y;
  _diag.z = diag.x;
  diag.x = _diag.x;
  off_diag.x = _off_diag.x;
  diag.y = _diag.y;
  off_diag.y = _off_diag.y;
  off_diag.z = _off_diag.z;
  diag.z = _diag.z;
}

template <typename TQ, typename TF3>
constexpr __device__ TQ jacobi_eigen_analysis(TF3 diag, TF3 off_diag) {
  TQ q{0, 0, 0, 1};
  constexpr int kJacobiEigenSteps = 14;
  for (int i = 0; i < kJacobiEigenSteps; ++i) {
    jacobi_conjugation(0, 1, 2, diag, off_diag, q);
    jacobi_conjugation(1, 2, 0, diag, off_diag, q);
    jacobi_conjugation(2, 0, 1, diag, off_diag, q);
  }
  return q;
}

template <typename TF>
__device__ __forceinline__ void swap_if(bool c, TF& x, TF& y) {
  TF z = x;
  x = c ? y : x;
  y = c ? z : y;
}

template <typename TF>
__device__ __forceinline__ void swap_if_not(bool c, TF& x, TF& y) {
  TF z = -x;
  x = c ? y : x;
  y = c ? z : y;
}

template <typename TF3>
constexpr __device__ void sort_singular(TF3& b0, TF3& b1, TF3& b2, TF3& v0,
                                        TF3& v1, TF3& v2) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  TF rho1 = length_sqr(b0.x, b1.x, b2.x);
  TF rho2 = length_sqr(b0.y, b1.y, b2.y);
  TF rho3 = length_sqr(b0.z, b1.z, b2.z);
  bool c;
  c = rho1 < rho2;
  swap_if_not(c, b0.x, b0.y);
  swap_if_not(c, v0.x, v0.y);
  swap_if_not(c, b1.x, b1.y);
  swap_if_not(c, v1.x, v1.y);
  swap_if_not(c, b2.x, b2.y);
  swap_if_not(c, v2.x, v2.y);
  swap_if(c, rho1, rho2);
  c = rho1 < rho3;
  swap_if_not(c, b0.x, b0.z);
  swap_if_not(c, v0.x, v0.z);
  swap_if_not(c, b1.x, b1.z);
  swap_if_not(c, v1.x, v1.z);
  swap_if_not(c, b2.x, b2.z);
  swap_if_not(c, v2.x, v2.z);
  swap_if(c, rho1, rho3);
  c = rho2 < rho3;
  swap_if_not(c, b0.y, b0.z);
  swap_if_not(c, v0.y, v0.z);
  swap_if_not(c, b1.y, b1.z);
  swap_if_not(c, v1.y, v1.z);
  swap_if_not(c, b2.y, b2.z);
  swap_if_not(c, v2.y, v2.z);
}

template <typename TF2, typename TF>
constexpr __device__ TF2 qr_givens(TF a1, TF a2) {
  // a1 = pivot point on diagonal
  // a2 = lower triangular entry we want to annihilate
  constexpr TF kSVDEpsilon = static_cast<TF>(1e-6);
  TF rho = sqrt(fma(a1, a1, +a2 * a2));
  TF2 g{fabs(a1) + fmax(rho, kSVDEpsilon), rho > kSVDEpsilon ? a2 : 0};
  bool b = a1 < 0;
  swap_if(b, g.y, g.x);
  return normalize(g);
}

template <typename TF3>
constexpr __device__ void qr_decompose(TF3& b0, TF3& b1, TF3& b2, TF3& q0,
                                       TF3& q1, TF3& q2, TF3& r0, TF3& r1,
                                       TF3& r2) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float2, double2> TF2;
  // first givens rotation (x,0,0,y)
  TF2 g1 = qr_givens<TF2>(b0.x, b1.x);
  TF a = fma(static_cast<TF>(-2), g1.y * g1.y, static_cast<TF>(1));
  TF b = 2 * g1.x * g1.y;
  // apply b = q' * b
  r0.x = fma(a, b0.x, b * b1.x);
  r0.y = fma(a, b0.y, b * b1.y);
  r0.z = fma(a, b0.z, b * b1.z);
  r1.x = fma(-b, b0.x, a * b1.x);
  r1.y = fma(-b, b0.y, a * b1.y);
  r1.z = fma(-b, b0.z, a * b1.z);
  r2.x = b2.x;
  r2.y = b2.y;
  r2.z = b2.z;
  // second givens rotation (x,0,-y,0)
  TF2 g2 = qr_givens<TF2>(r0.x, r2.x);
  a = fma(static_cast<TF>(-2), g2.y * g2.y, static_cast<TF>(1));
  b = 2 * g2.x * g2.y;
  // apply b = q' * b;
  b0.x = fma(a, r0.x, b * r2.x);
  b0.y = fma(a, r0.y, b * r2.y);
  b0.z = fma(a, r0.z, b * r2.z);
  b1.x = r1.x;
  b1.y = r1.y;
  b1.z = r1.z;
  b2.x = fma(-b, r0.x, a * r2.x);
  b2.y = fma(-b, r0.y, a * r2.y);
  b2.z = fma(-b, r0.z, a * r2.z);
  // third givens rotation (x,y,0,0)
  TF2 g3 = qr_givens<TF2>(b1.y, b2.y);
  a = fma(static_cast<TF>(-2), g3.y * g3.y, static_cast<TF>(1));
  b = 2 * g3.x * g3.y;
  // r is now set to desired value
  r0.x = b0.x;
  r0.y = b0.y;
  r0.z = b0.z;
  r1.x = fma(a, b1.x, b * b2.x);
  r1.y = fma(a, b1.y, b * b2.y);
  r1.z = fma(a, b1.z, b * b2.z);
  r2.x = fma(-b, b1.x, a * b2.x);
  r2.y = fma(-b, b1.y, a * b2.y);
  r2.z = fma(-b, b1.z, a * b2.z);
  // construct the cumulative rotation q=q1 * q2 * q3
  // the number of floating point operations for three quaternion
  // multiplications is more or less comparable to the explicit form of the
  // joined matrix. certainly more memory-efficient!
  TF sh12 = 2 * fma(g1.y, g1.y, static_cast<TF>(-0.5));
  TF sh22 = 2 * fma(g2.y, g2.y, static_cast<TF>(-0.5));
  TF sh32 = 2 * fma(g3.y, g3.y, static_cast<TF>(-0.5));
  q0.x = sh12 * sh22;
  q0.y = fma(4 * g2.x * g3.x, sh12 * g2.y * g3.y, 2 * g1.x * g1.y * sh32);
  q0.z = fma(4 * g1.x * g3.x, g1.y * g3.y, -2 * g2.x * sh12 * g2.y * sh32);

  q1.x = -2 * g1.x * g1.y * sh22;
  q1.y = fma(-8 * g1.x * g2.x * g3.x, g1.y * g2.y * g3.y, sh12 * sh32);
  q1.z = fma(-2 * g3.x, g3.y,
             4 * g1.y * fma(g3.x * g1.y, g3.y, g1.x * g2.x * g2.y * sh32));

  q2.x = 2 * g2.x * g2.y;
  q2.y = -2 * g3.x * sh22 * g3.y;
  q2.z = sh22 * sh32;
}

template <typename TF3>
constexpr __device__ void svd(TF3 const& diag, TF3 const& off_diag, TF3& u0,
                              TF3& u1, TF3& u2, TF3& s0, TF3& s1, TF3& s2,
                              TF3& v0, TF3& v1, TF3& v2) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float4, double4> TQ;
  TQ v_q = jacobi_eigen_analysis<TQ>(
      symmetric_matrix_product_diag(diag, off_diag),
      symmetric_matrix_product_off_diag(diag, off_diag));
  TF3 b0, b1, b2;
  quaternion_to_matrix(v_q, v0, v1, v2);
  matrix_multiply(diag, off_diag, v0, v1, v2, b0, b1, b2);
  sort_singular(b0, b1, b2, v0, v1, v2);
  qr_decompose(b0, b1, b2, u0, u1, u2, s0, s1, s2);
}

template <typename TF3>
__global__ void svd_kernel(Variable<1, TF3> diag, Variable<1, TF3> off_diag,
                           Variable<1, TF3> u0, Variable<1, TF3> u1,
                           Variable<1, TF3> u2, Variable<1, TF3> s0,
                           Variable<1, TF3> s1, Variable<1, TF3> s2,
                           Variable<1, TF3> v0, Variable<1, TF3> v1,
                           Variable<1, TF3> v2, U n) {
  forThreadMappedToElement(n, [&](U i) {
    svd(diag(i), off_diag(i), u0(i), u1(i), u2(i), s0(i), s1(i), s2(i), v0(i),
        v1(i), v2(i));
  });
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
  TF q = fabs(r) / cn<TF>().kernel_radius;
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
constexpr __device__ TF dist_gaussian_kernel(TF r, TF h) {
  return exp(-r * r / (h * h));
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

template <typename TF>
struct SqrtOperation {
  __device__ TF operator()(TF const& v) { return sqrt(v); }
};

template <typename M, typename TPrimitive>
struct SquaredDifferenceWeighted {
  __device__ TPrimitive operator()(thrust::tuple<M, M, TPrimitive> const& vvw) {
    M diff = thrust::get<0>(vvw) - thrust::get<1>(vvw);
    return thrust::get<2>(vvw) * length_sqr(diff);
  }
};

template <typename M, typename TPrimitive>
struct SquaredDifferenceYzWeighted {
  __device__ TPrimitive operator()(thrust::tuple<M, M, TPrimitive> const& vvw) {
    M diff = thrust::get<0>(vvw) - thrust::get<1>(vvw);
    return thrust::get<2>(vvw) * (diff.y * diff.y + diff.z * diff.z);
  }
};

template <typename TF>
struct KLDivergence {
  KLDivergence(U n_p, U n_q, TF q_lower_bound_arg)
      : n_p_inv(static_cast<TF>(1) / n_p),
        n_q_div_n_p(static_cast<TF>(n_q) / n_p),
        q_lower_bound(q_lower_bound_arg) {}
  __device__ TF operator()(thrust::tuple<U, U> const& freq_p_q) {
    TF freq_p = static_cast<TF>(thrust::get<0>(freq_p_q));
    TF freq_q = static_cast<TF>(thrust::get<1>(freq_p_q));
    return freq_p == 0
               ? 0
               : freq_p * n_p_inv *
                     log(freq_p / max(freq_q, q_lower_bound) * n_q_div_n_p);
  }
  TF n_p_inv;
  TF n_q_div_n_p;
  TF q_lower_bound;
};

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
constexpr __device__ TF3 index_to_position_in_fluid_cylinder(U p_i, TF radius,
                                                             TF particle_radius,
                                                             TF y_min,
                                                             TF y_max) {
  U sqrt_n;
  U n;
  U steps_y;
  TF diameter;
  get_fluid_cylinder_attr(radius, y_min, y_max, particle_radius, sqrt_n, n,
                          steps_y, diameter);

  U j = p_i % sqrt_n;
  U k = (p_i % n) / sqrt_n;
  U l = p_i / (n);
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
  return TF3{r * cos(theta), y_min + (l + 1) * diameter, r * sin(theta)};
}

template <typename TF>
__global__ void print_cn() {
  printf("kernel_radius = %f\n", cn<TF>().kernel_radius);
  printf("kernel_radius_sqr = %f\n", cn<TF>().kernel_radius_sqr);
  printf("cubic_kernel_k = %f\n", cn<TF>().cubic_kernel_k);
  printf("cubic_kernel_l = %f\n", cn<TF>().cubic_kernel_l);
  printf("cubic_kernel_zero = %f\n", cn<TF>().cubic_kernel_zero);
  printf("adhesion_kernel_k = %f\n", cn<TF>().adhesion_kernel_k);
  printf("cohesion_kernel_k = %f\n", cn<TF>().cohesion_kernel_k);
  printf("cohesion_kernel_c = %f\n", cn<TF>().cohesion_kernel_c);
  printf("particle_radius = %f\n", cn<TF>().particle_radius);
  printf("particle_vol = %f\n", cn<TF>().particle_vol);
  printf("particle_mass = %f\n", cn<TF>().particle_mass);
  printf("density0 = %f\n", cn<TF>().density0);
  printf("viscosity = %f\n", cn<TF>().viscosity);
  printf("boundary_viscosity = %f\n", cn<TF>().boundary_viscosity);
  printf("vorticity_coeff = %f\n", cn<TF>().vorticity_coeff);
  printf("boundary_vorticity_coeff = %f\n", cn<TF>().boundary_vorticity_coeff);
  printf("inertia_inverse = %f\n", cn<TF>().inertia_inverse);
  printf("viscosity_omega = %f\n", cn<TF>().viscosity_omega);
  printf("surface_tension_coeff = %f\n", cn<TF>().surface_tension_coeff);
  printf("surface_tension_boundary_coeff = %f\n",
         cn<TF>().surface_tension_boundary_coeff);
  printf("gravity = (%f, %f, %f)\n", cn<TF>().gravity.x, cn<TF>().gravity.y,
         cn<TF>().gravity.z);
  printf("boundary_epsilon = %f\n", cn<TF>().boundary_epsilon);
  printf("dfsph_factor_epsilon = %f\n", cn<TF>().dfsph_factor_epsilon);
  printf("contact_tolerance = %f\n", cn<TF>().contact_tolerance);
  printf("wrap_length = %f\n", cn<TF>().wrap_length);
  printf("wrap_min = %f\n", cn<TF>().wrap_min);
  printf("wrap_max = %f\n", cn<TF>().wrap_max);
  printf("max_num_particles_per_cell = %u\n", cni.max_num_particles_per_cell);
  printf("grid_res = (%u, %u, %u)\n", cni.grid_res.x, cni.grid_res.y,
         cni.grid_res.z);
  printf("grid_offset = (%d, %d, %d)\n", cni.grid_offset.x, cni.grid_offset.y,
         cni.grid_offset.z);
  printf("max_num_neighbors_per_particle = %u\n",
         cni.max_num_neighbors_per_particle);
  printf("num_boundaries = %u\n", cni.num_boundaries);
  printf("max_num_contacts = %u\n", cni.max_num_contacts);
}
template <typename TF3, typename TF>
__global__ void create_fluid_cylinder(Variable<1, TF3> particle_x,
                                      U num_particles, U offset, TF radius,
                                      TF particle_radius, TF y_min, TF y_max) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_x(offset + p_i) = index_to_position_in_fluid_cylinder<TF3>(
        p_i, radius, particle_radius, y_min, y_max);
  });
}

template <typename TF3, typename TF>
__global__ void create_fluid_cylinder_internal(
    Variable<1, TF3> particle_x, Variable<1, U> internal_encoded_sorted,
    U num_particles, U offset, TF radius, TF particle_radius, TF y_min,
    TF y_max) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_x(offset + p_i) = index_to_position_in_fluid_cylinder<TF3>(
        internal_encoded_sorted(p_i), radius, particle_radius, y_min, y_max);
  });
}

template <typename TQ, typename TF3, typename TF, typename TDistance>
__global__ void compute_fluid_cylinder_internal(Variable<1, U> internal_encoded,
                                                const TDistance distance,
                                                TF sign, TF3 rigid_x,
                                                TQ rigid_q, U num_positions,
                                                TF radius, TF particle_radius,
                                                TF y_min, TF y_max) {
  forThreadMappedToElement(num_positions, [&](U p_i) {
    TF3 x = index_to_position_in_fluid_cylinder<TF3>(
        p_i, radius, particle_radius, y_min, y_max);

    TF d = distance.signed_distance(rotate_using_quaternion(
               x - rigid_x, quaternion_conjugate(rigid_q))) *
           sign;
    internal_encoded(p_i) =
        (d < cn<TF>().kernel_radius or internal_encoded(p_i) == UINT_MAX)
            ? UINT_MAX
            : p_i;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_fluid_cylinder_internal(
    Variable<1, U> internal_encoded, Variable<1, TF> distance_nodes,
    TF3 domain_min, TF3 domain_max, U3 resolution, TF3 cell_size, TF sign,
    TF3 rigid_x, TQ rigid_q, U num_positions, TF radius, TF particle_radius,
    TF y_min, TF y_max) {
  forThreadMappedToElement(num_positions, [&](U p_i) {
    TF3 x = index_to_position_in_fluid_cylinder<TF3>(
        p_i, radius, particle_radius, y_min, y_max);
    TF distance =
        interpolate_distance_without_intermediates(
            &distance_nodes, domain_min, domain_max, resolution, cell_size,
            rotate_using_quaternion(x - rigid_x,
                                    quaternion_conjugate(rigid_q))) *
        sign;
    internal_encoded(p_i) =
        (distance < cn<TF>().kernel_radius or internal_encoded(p_i) == UINT_MAX)
            ? UINT_MAX
            : p_i;
  });
}

template <typename TF3, typename TF>
__device__ __host__ inline void get_fluid_block_attr(int mode,
                                                     TF3 const& box_min,
                                                     TF3 const& box_max,
                                                     TF particle_radius,
                                                     I3& steps) {
  TF diameter = particle_radius * 2;
  TF xshift, yshift;
  TF3 diff = box_max - box_min;
  if (mode == 0) {
    xshift = diameter;
    yshift = diameter;
  } else if (mode == 1) {
    xshift = diameter;
    yshift = sqrt(static_cast<TF>(3)) * particle_radius;
    diff.x -= diameter;
    diff.z -= diameter;
  } else {
    xshift = kHcpStep0<TF> * particle_radius;
    yshift = kHcpStep1<TF> * particle_radius;
    diff.x -= xshift;
    diff.z -= diameter;
  }
  steps.x = static_cast<I>(diff.x / xshift + static_cast<TF>(0.5)) - 1;
  steps.y = static_cast<I>(diff.y / yshift + static_cast<TF>(0.5)) - 1;
  steps.z = static_cast<I>(diff.z / diameter + static_cast<TF>(0.5)) - 1;
}

template <typename TF3, typename TF>
constexpr __device__ TF3 index_to_position_in_fluid_block(U p_i, int mode,
                                                          TF3 const& box_min,
                                                          I3 const& steps,
                                                          TF particle_radius) {
  I j = (p_i % (steps.x * steps.z)) / steps.z;
  I k = p_i / (steps.x * steps.z);
  I l = p_i % steps.z;
  TF3 currPos = TF3{particle_radius * 2 * j, particle_radius * 2 * k,
                    particle_radius * 2 * l};
  TF3 start = box_min + make_vector<TF3>(particle_radius * 2);
  if (mode == 1) {
    if (k % 2 == 0) {
      currPos.z += particle_radius;
    } else {
      currPos.x += particle_radius;
    }
  } else if (mode == 2) {
    currPos.z = particle_radius * (l * 2 + (j + k) % 2);
    currPos.x =
        particle_radius * kHcpStep0<TF> * (j + static_cast<TF>(k % 2) / 3);
    currPos.y = particle_radius * kHcpStep1<TF> * k;
  }
  return currPos + start;
}

template <typename TF3, typename TF>
__global__ void create_fluid_block(Variable<1, TF3> particle_x, U num_particles,
                                   U offset, TF particle_radius, int mode,
                                   TF3 box_min, TF3 box_max) {
  I3 steps;
  get_fluid_block_attr(mode, box_min, box_max, particle_radius, steps);
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_x(offset + p_i) = index_to_position_in_fluid_block(
        p_i, mode, box_min, steps, particle_radius);
  });
}

template <typename TF3, typename TF>
__global__ void create_custom_beads_internal(
    Variable<1, TF3> particle_x, Variable<1, TF3> ref_x,
    Variable<1, U> internal_encoded_sorted, U num_particles, U offset) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_x(offset + p_i) = ref_x(internal_encoded_sorted(p_i));
  });
}

template <typename TF>
__global__ void create_custom_beads_scalar_internal(
    Variable<1, TF> particle_scalar, Variable<1, TF> ref_scalar,
    Variable<1, U> internal_encoded_sorted, U num_particles, U offset) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_scalar(offset + p_i) = ref_scalar(internal_encoded_sorted(p_i));
  });
}

template <typename TF3, typename TF>
__global__ void create_fluid_block_internal(
    Variable<1, TF3> particle_x, Variable<1, U> internal_encoded_sorted,
    U num_particles, U offset, TF particle_radius, int mode, TF3 box_min,
    TF3 box_max) {
  I3 steps;
  get_fluid_block_attr(mode, box_min, box_max, particle_radius, steps);
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_x(offset + p_i) = index_to_position_in_fluid_block(
        internal_encoded_sorted(p_i), mode, box_min, steps, particle_radius);
  });
}

template <typename TQ, typename TF3, typename TF, typename TDistance>
__global__ void compute_fluid_block_internal(Variable<1, U> internal_encoded,
                                             const TDistance distance, TF sign,
                                             TF3 rigid_x, TQ rigid_q,
                                             U num_positions,
                                             TF particle_radius, int mode,
                                             TF3 box_min, TF3 box_max) {
  I3 steps;
  get_fluid_block_attr(mode, box_min, box_max, particle_radius, steps);
  forThreadMappedToElement(num_positions, [&](U p_i) {
    TF3 x = index_to_position_in_fluid_block(p_i, mode, box_min, steps,
                                             particle_radius);

    TF d = distance.signed_distance(rotate_using_quaternion(
               x - rigid_x, quaternion_conjugate(rigid_q))) *
           sign;
    internal_encoded(p_i) =
        (d < cn<TF>().kernel_radius or internal_encoded(p_i) == UINT_MAX)
            ? UINT_MAX
            : p_i;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_custom_beads_internal(Variable<1, U> internal_encoded,
                                              Variable<1, TF3> bead_x,
                                              Variable<1, TF> distance_nodes,
                                              TF3 domain_min, TF3 domain_max,
                                              U3 resolution, TF3 cell_size,
                                              TF sign, TF3 rigid_x, TQ rigid_q,
                                              U num_positions) {
  forThreadMappedToElement(num_positions, [&](U p_i) {
    TF3 x = bead_x(p_i);

    TF distance =
        interpolate_distance_without_intermediates(
            &distance_nodes, domain_min, domain_max, resolution, cell_size,
            rotate_using_quaternion(x - rigid_x,
                                    quaternion_conjugate(rigid_q))) *
        sign;
    internal_encoded(p_i) =
        (distance < cn<TF>().kernel_radius or internal_encoded(p_i) == UINT_MAX)
            ? UINT_MAX
            : p_i;
  });
}

template <typename TQ, typename TF3, typename TF, typename TDistance>
__global__ void compute_custom_beads_internal(Variable<1, U> internal_encoded,
                                              Variable<1, TF3> bead_x,
                                              const TDistance distance, TF sign,
                                              TF3 rigid_x, TQ rigid_q,
                                              U num_positions) {
  forThreadMappedToElement(num_positions, [&](U p_i) {
    TF3 x = bead_x(p_i);

    TF d = distance.signed_distance(rotate_using_quaternion(
               x - rigid_x, quaternion_conjugate(rigid_q))) *
           sign;
    internal_encoded(p_i) =
        (d < cn<TF>().kernel_radius or internal_encoded(p_i) == UINT_MAX)
            ? UINT_MAX
            : p_i;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_fluid_block_internal(
    Variable<1, U> internal_encoded, Variable<1, TF> distance_nodes,
    TF3 domain_min, TF3 domain_max, U3 resolution, TF3 cell_size, TF sign,
    TF3 rigid_x, TQ rigid_q, U num_positions, TF particle_radius, int mode,
    TF3 box_min, TF3 box_max) {
  I3 steps;
  get_fluid_block_attr(mode, box_min, box_max, particle_radius, steps);
  forThreadMappedToElement(num_positions, [&](U p_i) {
    TF3 x = index_to_position_in_fluid_block(p_i, mode, box_min, steps,
                                             particle_radius);

    TF distance =
        interpolate_distance_without_intermediates(
            &distance_nodes, domain_min, domain_max, resolution, cell_size,
            rotate_using_quaternion(x - rigid_x,
                                    quaternion_conjugate(rigid_q))) *
        sign;
    internal_encoded(p_i) =
        (distance < cn<TF>().kernel_radius or internal_encoded(p_i) == UINT_MAX)
            ? UINT_MAX
            : p_i;
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
__device__ TF interpolate(Variable<1, TF>* nodes, U* cells, TF* N) {
  TF phi = 0;
  for (U j = 0; j < 32; ++j) {
    TF c = (*nodes)(cells[j]);
    phi += c * N[j];
  }
  return phi;
}

template <typename TF3, typename TF>
__device__ TF interpolate_distance_without_intermediates(Variable<1, TF>* nodes,
                                                         TF3 domain_min,
                                                         TF3 domain_max,
                                                         U3 resolution,
                                                         TF3 cell_size, TF3 x) {
  I3 ipos;
  TF3 inner_x;
  TF N[32];
  U cells[32];
  TF d = kFMax<TF>;
  resolve(domain_min, domain_max, resolution, cell_size, x, &ipos, &inner_x);
  if (ipos.x >= 0) {
    get_shape_function(inner_x, N);
    get_cells(resolution, ipos, cells);
    d = interpolate(nodes, cells, N);
  }
  return d;
}

template <typename TF3, typename TF>
__global__ void update_volume_field(Variable<1, TF> volume_nodes,
                                    Variable<1, TF> distance_nodes,
                                    TF3 domain_min, TF3 domain_max,
                                    U3 resolution, TF3 cell_size, U num_nodes,
                                    TF sign) {
  forThreadMappedToElement(num_nodes, [&](U l) {
    TF3 x = index_to_node_position(domain_min, resolution, cell_size, l);
    TF dist = distance_nodes(l);
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
              x + integrand_parameter);
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
    volume_nodes(l) = sum * cn<TF>().kernel_radius_sqr * cn<TF>().kernel_radius;
  });
}

template <typename TF3, typename TF, typename TDistance>
__global__ void update_volume_field(Variable<1, TF> volume_nodes,
                                    const TDistance distance, TF3 domain_min,
                                    U3 resolution, TF3 cell_size, U num_nodes,
                                    TF sign) {
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
    volume_nodes(l) = sum * cn<TF>().kernel_radius_sqr * cn<TF>().kernel_radius;
  });
}

template <typename TQ, typename TF3, typename TF>
__device__ TF compute_volume_and_boundary_x(
    Variable<1, TF>* volume_nodes, Variable<1, TF>* distance_nodes,
    TF3 domain_min, TF3 domain_max, U3 resolution, TF3 cell_size, TF sign,
    TF3& x, TF3& rigid_x, TQ& rigid_q, TF3* boundary_xj, TF3* xi_bxj, TF* d) {
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
  *d = 0;

  resolve(domain_min, domain_max, resolution, cell_size, local_xi, &ipos,
          &inner_x);
  if (ipos.x >= 0) {
    get_shape_function_and_gradient(inner_x, N, dN0, dN1, dN2);
    get_cells(resolution, ipos, cells);
    *d = interpolate_and_derive(distance_nodes, &cell_size, cells, N, dN0, dN1,
                                dN2, &normal) *
         sign;
    normal = rotate_using_quaternion(normal * sign, rigid_q);
    TF nl2 = length_sqr(normal);
    normal *= (nl2 > 0 ? rsqrt(nl2) : 0);
    *xi_bxj = (*d) * normal;
    *boundary_xj = x - (*xi_bxj);
    boundary_volume = interpolate(volume_nodes, cells, N);
  }
  return boundary_volume;
}

template <U wrap, typename TQ, typename TF3, typename TF, typename TDistance>
__device__ TF compute_volume_and_boundary_x_analytic(
    Variable<1, TF>* volume_nodes, TDistance const& distance, TF3 domain_min,
    TF3 domain_max, U3 resolution, TF3 cell_size, TF sign, TF3& x, TF3& rigid_x,
    TQ& rigid_q, TF3* boundary_xj, TF3* xi_bxj, TF* d) {
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

  *d = distance.signed_distance(local_xi) * sign;
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
    boundary_volume = interpolate(volume_nodes, cells, N);
  }

  return boundary_volume;
}

// gradient must be initialized to zero
template <typename TF3, typename TF>
__device__ TF interpolate_and_derive(Variable<1, TF>* nodes, TF3* cell_size,
                                     U* cells, TF* N, TF* dN0, TF* dN1, TF* dN2,
                                     TF3* gradient) {
  TF phi = 0;
  *gradient = TF3{0};
  for (U j = 0; j < 32; ++j) {
    TF c = (*nodes)(cells[j]);
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
                                         U3 resolution, TF3 cell_size, TF sign,
                                         TF tolerance, TF3 x, TF3* normal) {
  TF dist = sign;  // in extreme condition when the vertex is outside the whole
                   // distance map, finding the correct distance is impossible.
                   // Only report the existence of collision.
  I3 ipos;
  TF3 inner_x;
  TF N[32];
  TF dN0[32];
  TF dN1[32];
  TF dN2[32];
  U cells[32];
  TF3 normal_tmp{0, 1, 0};
  *normal = TF3{0};

  resolve(domain_min, domain_max, resolution, cell_size, x, &ipos, &inner_x);
  if (ipos.x >= 0) {
    get_shape_function_and_gradient(inner_x, N, dN0, dN1, dN2);
    get_cells(resolution, ipos, cells);
    dist = sign * interpolate_and_derive(distance_nodes, &cell_size, cells, N,
                                         dN0, dN1, dN2, &normal_tmp) -
           tolerance;
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
                                     Variable<1, U> grid_anomaly,
                                     U num_particles, U offset) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  forThreadMappedToElement(num_particles, [&](U p_i0) {
    U p_i = p_i0 + offset;
    TF3 x_i = particle_x(p_i);
    I3 ipos = get_ipos(x_i);
    U pid_insert_index;
    if (within_grid(ipos)) {
      pid_insert_index = atomicAdd(&pid_length(ipos), 1);
      // NOTE: pack x_i and p_i together to avoid scatter-read from particle_x
      // in make_neighbor_list
      TQ pid_entry{x_i.x, x_i.y, x_i.z, 0};
      reinterpret_cast<U&>(pid_entry.w) = p_i;
      pid(ipos, min(cni.max_num_particles_per_cell - 1, pid_insert_index)) =
          pid_entry;
      if (pid_insert_index == cni.max_num_particles_per_cell) {
        grid_anomaly(1) = 1;
      }
    } else {
      grid_anomaly(0) = 1;
    }
  });
}

template <U wrap, typename TQ, typename TF3>
__global__ void make_neighbor_list(Variable<1, TF3> sample_x,
                                   Variable<4, TQ> pid,
                                   Variable<3, U> pid_length,
                                   Variable<2, TQ> sample_neighbors,
                                   Variable<1, U> sample_num_neighbors,
                                   Variable<1, U> grid_anomaly, U num_samples) {
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
            sample_neighbors(p_i, min(cni.max_num_neighbors_per_particle - 1,
                                      num_neighbors)) = neighbor_entry;
            num_neighbors += 1;
          }
        }
      }
    }
    sample_num_neighbors(p_i) =
        min(cni.max_num_neighbors_per_particle, num_neighbors);
    if (num_neighbors > cni.max_num_neighbors_per_particle) {
      grid_anomaly(2) = 1;
    }
  });
}

template <U wrap, typename TQ, typename TF3>
__global__ void make_bead_pellet_neighbor_list(
    Variable<1, TF3> sample_x, Variable<4, TQ> pid, Variable<3, U> pid_length,
    Variable<2, TQ> sample_bead_neighbors,
    Variable<1, U> sample_num_bead_neighbors,
    Variable<2, TQ> sample_pellet_neighbors,
    Variable<1, U> sample_num_pellet_neighbors, Variable<1, U> grid_anomaly,
    U max_num_beads, U num_samples, U offset) {
  typedef std::conditional_t<std::is_same_v<TF3, float3>, float, double> TF;
  forThreadMappedToElement(num_samples, [&](U p_i0) {
    U p_i = p_i0 + offset;
    TF3 x = sample_x(p_i);
    I3 ipos = get_ipos(x);
    U num_bead_neighbors = 0;
    U num_pellet_neighbors = 0;
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
            if (p_j < max_num_beads) {
              TQ neighbor_entry{xixj.x, xixj.y, xixj.z, 0};
              reinterpret_cast<U&>(neighbor_entry.w) = p_j;
              sample_bead_neighbors(p_i,
                                    min(cni.max_num_neighbors_per_particle - 1,
                                        num_bead_neighbors)) = neighbor_entry;
              num_bead_neighbors += 1;
            } else {
              TQ neighbor_entry{xixj.x, xixj.y, xixj.z, 0};
              reinterpret_cast<U&>(neighbor_entry.w) = p_j;
              sample_pellet_neighbors(
                  p_i, min(cni.max_num_neighbors_per_particle - 1,
                           num_pellet_neighbors)) = neighbor_entry;
              num_pellet_neighbors += 1;
            }
          }
        }
      }
    }
    sample_num_bead_neighbors(p_i) =
        min(cni.max_num_neighbors_per_particle, num_bead_neighbors);
    sample_num_pellet_neighbors(p_i) =
        min(cni.max_num_neighbors_per_particle, num_pellet_neighbors);
    if (num_bead_neighbors > cni.max_num_neighbors_per_particle ||
        num_pellet_neighbors > cni.max_num_neighbors_per_particle) {
      grid_anomaly(2) = 1;
    }
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

template <U wrap, typename TQ, typename TF3, typename TF, typename TDistance>
__global__ void compute_particle_boundary_analytic(
    Variable<1, TF> volume_nodes, const TDistance distance, TF3 rigid_x,
    TQ rigid_q, U boundary_id, TF3 domain_min, TF3 domain_max, U3 resolution,
    TF3 cell_size, TF sign, Variable<1, TF3> particle_x,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 boundary_xj, xi_bxj;
    TF d;
    TF boundary_volume = compute_volume_and_boundary_x_analytic<wrap>(
        &volume_nodes, distance, domain_min, domain_max, resolution, cell_size,
        sign, particle_x(p_i), rigid_x, rigid_q, &boundary_xj, &xi_bxj, &d);
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
    TF3 cell_size, TF sign, Variable<1, TF3> particle_x,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 boundary_xj, xi_bxj;
    TF d;
    TF boundary_volume = compute_volume_and_boundary_x(
        &volume_nodes, &distance_nodes, domain_min, domain_max, resolution,
        cell_size, sign, particle_x(p_i), rigid_x, rigid_q, &boundary_xj,
        &xi_bxj, &d);
    TF3 kernel_grad = displacement_cubic_kernel_grad(xi_bxj);
    particle_boundary(boundary_id, p_i) =
        TQ{boundary_xj.x, boundary_xj.y, boundary_xj.z, boundary_volume};
    particle_boundary_kernel(boundary_id, p_i) =
        boundary_volume *
        TQ{kernel_grad.x, kernel_grad.y, kernel_grad.z, dist_cubic_kernel(d)};
  });
}

template <typename TQ>
__global__ void compute_particle_boundary_with_pellets(
    Variable<1, TQ> particle_boundary_kernel_combined,
    Variable<2, TQ> sample_pellet_neighbors,
    Variable<1, U> sample_num_pellet_neighbors, U num_particles) {
  typedef std::conditional_t<std::is_same_v<TQ, float4>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TQ, float4>, float, double> TF;
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 boundary_kernel_grad{0};
    TF boundary_kernel = 0;
    for (U neighbor_id = 0; neighbor_id < sample_num_pellet_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(sample_pellet_neighbors(p_i, neighbor_id), xixj, p_j);
      boundary_kernel += displacement_cubic_kernel(xixj);
      boundary_kernel_grad += displacement_cubic_kernel_grad(xixj);
    }
    particle_boundary_kernel_combined(p_i) =
        cn<TF>().particle_vol * TQ{boundary_kernel_grad.x,
                                   boundary_kernel_grad.y,
                                   boundary_kernel_grad.z, boundary_kernel};
  });
}

template <typename TQ, typename TF>
__global__ void compute_density(Variable<2, TQ> particle_neighbors,
                                Variable<1, U> particle_num_neighbors,
                                Variable<1, TF> particle_density,
                                Variable<2, TQ> particle_boundary_kernel,
                                U num_particles) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF density = cn<TF>().particle_vol * cn<TF>().cubic_kernel_zero;
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

template <typename TQ, typename TF>
__global__ void compute_density_with_pellets(
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary_neighbors,
    Variable<1, U> particle_num_boundary_neighbors,
    Variable<1, TF> particle_density, U num_particles) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF density = cn<TF>().cubic_kernel_zero;
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      extract_displacement(particle_neighbors(p_i, neighbor_id), xixj);
      density += displacement_cubic_kernel(xixj);
    }
    for (U neighbor_id = 0; neighbor_id < particle_num_boundary_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      U p_j;
      extract_pid(particle_boundary_neighbors(p_i, neighbor_id), xixj, p_j);
      density += displacement_cubic_kernel(xixj);
    }
    particle_density(p_i) = density * cn<TF>().particle_mass;
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

template <typename TQ, typename TF3, typename TF>
__global__ void sample_position_density_with_pellets(
    Variable<1, TF3> sample_x, Variable<2, TQ> sample_neighbors,
    Variable<1, U> sample_num_neighbors,
    Variable<2, TQ> sample_pellet_neighbors,
    Variable<1, U> sample_num_pellet_neighbors, Variable<1, TF> sample_density,
    U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF density = 0;
    TF3 x_i = sample_x(p_i);
    for (U neighbor_id = 0; neighbor_id < sample_num_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      extract_displacement(sample_neighbors(p_i, neighbor_id), xixj);
      density += displacement_cubic_kernel(xixj);
    }
    for (U neighbor_id = 0; neighbor_id < sample_num_pellet_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      extract_displacement(sample_pellet_neighbors(p_i, neighbor_id), xixj);
      density += displacement_cubic_kernel(xixj);
    }
    sample_density(p_i) = density * cn<TF>().particle_mass;
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

template <typename TQ, typename TF3, typename TF>
__global__ void compute_viscosity_with_pellets(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, Variable<1, TF3> particle_a,
    Variable<2, TF3> particle_force, Variable<2, TF3> particle_torque,
    Variable<2, TQ> particle_boundary_neighbors,
    Variable<1, U> particle_num_boundary_neighbors,
    Variable<1, U> pellet_id_to_rigid_id, Variable<1, TF3> rigid_x,
    U max_num_beads, U num_particles) {
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
    for (U neighbor_id = 0; neighbor_id < particle_num_boundary_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_boundary_neighbors(p_i, neighbor_id), xixj, p_j);
      TF3 a = cn<TF>().particle_vol * cn<TF>().boundary_viscosity *
              dot(xixj, displacement_cubic_kernel_grad(xixj)) /
              (length_sqr(xixj) +
               static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
              (v_i - particle_v(p_j));
      da += a;

      TF3 pellet_x = x_i - xixj;
      U boundary_id = pellet_id_to_rigid_id(p_j - max_num_beads);
      TF3 r_x = rigid_x(boundary_id);
      TF3 force = -cn<TF>().particle_mass * a;
      particle_force(boundary_id, p_i) += force;
      particle_torque(boundary_id, p_i) += cross(pellet_x - r_x, force);
    }
    particle_a(p_i) += da;
  });
}

// Micropolar Model
template <typename TQ, typename TF3, typename TF>
__global__ void compute_micropolar_vorticity(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<1, TF3> particle_omega,
    Variable<1, TF3> particle_a, Variable<1, TF3> particle_angular_acceleration,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TF3> particle_force, Variable<2, TF3> particle_torque,
    Variable<2, TQ> particle_boundary, Variable<2, TQ> particle_boundary_kernel,
    Variable<1, TF3> rigid_x, Variable<1, TF3> rigid_v,
    Variable<1, TF3> rigid_omega, TF dt, U num_particles) {
  // TODO: remove dt
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v_i = particle_v(p_i);
    TF3 omega_i = particle_omega(p_i);
    TF density_i = particle_density(p_i);

    TF3 da{0};
    TF3 dangular_acc = -2 * cn<TF>().vorticity_coeff * omega_i;

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      TF3 grad_w = displacement_cubic_kernel_grad(xixj);

      TF density_j = particle_density(p_j);
      TF3 omega_j = particle_omega(p_j);
      TF3 omega_ij = omega_i - omega_j;

      dangular_acc +=  // laplacian omega (why negative sign in SPlisHSPlasH?)
          -cn<TF>().viscosity_omega * cn<TF>().particle_mass / density_j *
              dot(xixj, grad_w) /
              (length_sqr(xixj) +
               static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
              omega_ij +
          // curl of velocity (symmetric formula)
          density_i * cn<TF>().vorticity_coeff * cn<TF>().particle_mass *
              cross(v_i / (density_i * density_i) +
                        particle_v(p_j) / (density_j * density_j),
                    grad_w);
      // curl of microrotation field (symmetric formula)
      da += cross(
          omega_i / (density_i * density_i) + omega_j / (density_j * density_j),
          grad_w);
      // // curl of microrotation field (difference formula)
      // da += cross(omega_ij, grad_w)/ density_j;
    }
    da *= density_i * cn<TF>().vorticity_coeff * cn<TF>().particle_mass;
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = particle_boundary(boundary_id, p_i);
      TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
      TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
                 rigid_v(boundary_id);
      TF3 x_ib = x_i - bx_j;
      TF3 omega_ij = omega_i;  // TODO: omegaj not implemented in SPlisHSPlasH
      // curl of microrotation_field (difference formula)
      TF3 a = cn<TF>().boundary_vorticity_coeff * cross(omega_ij, grad_wvol);
      da += a;
      dangular_acc +=
          // laplacian omega
          cn<TF>().viscosity_omega * dot(x_ib, grad_wvol) /
              (length_sqr(x_ib) +
               static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
              omega_ij +
          // curl of velocity (difference formula)
          cn<TF>().boundary_vorticity_coeff * cross(v_i - velj, grad_wvol);
      TF3 force = -cn<TF>().particle_mass * a;
      particle_force(boundary_id, p_i) += force;
      particle_torque(boundary_id, p_i) +=
          cross(bx_j - rigid_x(boundary_id), force);
    }

    particle_a(p_i) += da;
    particle_angular_acceleration(p_i) =
        dangular_acc * cn<TF>().inertia_inverse;
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

template <typename TQ, typename TF3, typename TF, typename TDistance>
__global__ void compute_cohesion_adhesion_displacement(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<1, TF3> particle_dx, const TDistance distance, TF sign,
    TF cohesion, TF adhesion, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF density_i = particle_density(p_i);
    TF3 dx{0};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);

      dx -= cohesion * cn<TF>().particle_mass / particle_density(p_j) *
            (1 - cn<TF>().particle_radius * 2 * rsqrt(length_sqr(xixj))) *
            displacement_cubic_kernel(xixj) * xixj;
    }

    TF dist = distance.signed_distance(x_i) * sign - cn<TF>().contact_tolerance;
    TF3 normal = distance.gradient(x_i, static_cast<TF>(1e-3)) * sign;
    TF normal_norm_sqr = length_sqr(normal);
    if (normal_norm_sqr >= static_cast<TF>(1e-7)) {
      dx -= adhesion * cn<TF>().particle_mass / density_i *
            dist_cubic_kernel(dist) * (dist * normalize(normal));
    }
    particle_dx(p_i) = dx;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_cohesion_adhesion_displacement(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<1, TF3> particle_dx, Variable<1, TF> distance_nodes,
    TF3 domain_min, TF3 domain_max, U3 resolution, TF3 cell_size, TF sign,
    TF cohesion, TF adhesion, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF density_i = particle_density(p_i);
    TF3 dx{0};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);

      dx -= cohesion * cn<TF>().particle_mass / particle_density(p_j) *
            (1 - cn<TF>().particle_radius * 2 * rsqrt(length_sqr(xixj))) *
            displacement_cubic_kernel(xixj) * xixj;
    }

    // for resolve
    I3 ipos;
    TF3 inner_x;
    TF N[32];
    TF dN0[32];
    TF dN1[32];
    TF dN2[32];
    U cells[32];
    resolve(domain_min, domain_max, resolution, cell_size, x_i, &ipos,
            &inner_x);
    if (ipos.x >= 0) {
      get_shape_function_and_gradient(inner_x, N, dN0, dN1, dN2);
      get_cells(resolution, ipos, cells);
      TF3 normal;
      TF dist = interpolate_and_derive(&distance_nodes, &cell_size, cells, N,
                                       dN0, dN1, dN2, &normal) *
                sign;
      TF nl2 = length_sqr(normal);
      normal *= sign * (nl2 > 0 ? rsqrt(nl2) : 0);
      dx -= adhesion * cn<TF>().particle_mass / density_i *
            dist_cubic_kernel(dist) * (dist * normalize(normal));
    }
    particle_dx(p_i) = dx;
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
// MLS pressure extrapolation
template <typename TQ, typename TF3, typename TF>
__global__ void extrapolate_pressure_to_pellet(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_pressure,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    U max_num_beads, U num_pellets) {
  forThreadMappedToElement(num_pellets, [&](U p_i0) {
    U p_i = p_i0 + max_num_beads;
    TF w_sum = 0;
    TF3 neighbor_center{0};
    U num_neighbors = particle_num_neighbors(p_i);

    for (U neighbor_id = 0; neighbor_id < num_neighbors; ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      TF w = displacement_cubic_kernel(xixj) * cn<TF>().particle_vol;
      // NOTE: assume the same rest density
      w_sum += w;
      // TODO: optimize computation by using xixj instead of xj?
      neighbor_center += w * particle_x(p_j);
    }
    neighbor_center /= w_sum;
    TF alpha = 0;
    TF3 diag{0};
    TF3 off_diag{0};
    TF3 pressure_x_sum{0};

    for (U neighbor_id = 0; neighbor_id < num_neighbors; ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      TF3 x_j = particle_x(p_j);
      TF3 x_j_t = x_j - neighbor_center;
      TF w = displacement_cubic_kernel(xixj) * cn<TF>().particle_vol;
      TF w_pressure_j = w * particle_pressure(p_j);
      alpha += w_pressure_j;
      pressure_x_sum += w_pressure_j * x_j_t;
      diag += w * x_j_t * x_j_t;
      off_diag +=
          w * TF3{x_j_t.x * x_j_t.y, x_j_t.x * x_j_t.z, x_j_t.y * x_j_t.z};
    }
    alpha /= w_sum;
    TF3 u0, u1, u2;
    TF3 s0, s1, s2;
    TF3 v0, v1, v2;
    svd(diag, off_diag, u0, u1, u2, s0, s1, s2, v0, v1, v2);
    constexpr TF kSingularEpsilon = static_cast<TF>(1e-2);
    TF3 s_inv{fabs(s0.x) > kSingularEpsilon ? 1 / s0.x : 0,
              fabs(s1.y) > kSingularEpsilon ? 1 / s1.y : 0,
              fabs(s2.z) > kSingularEpsilon ? 1 / s2.z : 0};
    TF3 inv0, inv1, inv2;
    TF3 sinv_ut0 = s_inv.x * TF3{u0.x, u1.x, u2.x};
    TF3 sinv_ut1 = s_inv.y * TF3{u0.y, u1.y, u2.y};
    TF3 sinv_ut2 = s_inv.z * TF3{u0.z, u1.z, u2.z};
    matrix_multiply(v0, v1, v2, sinv_ut0, sinv_ut1, sinv_ut2, inv0, inv1, inv2);

    TF3 plane_param{dot(inv0, pressure_x_sum), dot(inv1, pressure_x_sum),
                    dot(inv2, pressure_x_sum)};
    TF m_det = diag.x * (diag.y * diag.z - off_diag.z * off_diag.z) -
               off_diag.x * (off_diag.x * diag.z - off_diag.z * off_diag.y) +
               off_diag.y * (off_diag.x * off_diag.z - diag.y * off_diag.y);
    TF pressure = alpha + dot(plane_param, particle_x(p_i) - neighbor_center);
    particle_pressure(p_i) = fmax(pressure, static_cast<TF>(0));
  });
}
// IISPH
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
      ai -= cn<TF>().particle_mass * (dpi + dpj) *
            displacement_cubic_kernel_grad(xixj);
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = particle_boundary(boundary_id, p_i);
      TQ boundary_kernel = particle_boundary_kernel(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);
      TF3 a = cn<TF>().density0 * dpi * grad_wvol;
      TF3 force = cn<TF>().particle_mass * a;
      ai -= a;
      particle_force(boundary_id, p_i) += force;
      particle_torque(boundary_id, p_i) +=
          cross(bx_j - rigid_x(boundary_id), force);
    }
    particle_pressure_accel(p_i) = cn<TF>().density0 * ai;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_pressure_accels_with_pellets(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_pressure, Variable<1, TF3> particle_pressure_accel,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TF3> particle_force, Variable<2, TF3> particle_torque,
    Variable<2, TQ> particle_boundary_neighbors,
    Variable<1, U> particle_num_boundary_neighbors,
    Variable<1, U> pellet_id_to_rigid_id, Variable<1, TF3> rigid_x,
    U max_num_beads, U num_particles) {
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
      ai -= cn<TF>().particle_mass * (dpi + dpj) *
            displacement_cubic_kernel_grad(xixj);
    }
    for (U neighbor_id = 0; neighbor_id < particle_num_boundary_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_boundary_neighbors(p_i, neighbor_id), xixj, p_j);
      TF3 x_j = x_i - xixj;
      U boundary_id = pellet_id_to_rigid_id(p_j - max_num_beads);

      TF densityj = cn<TF>().density0;
      TF densityj2 = densityj * densityj;
      TF dpj = particle_pressure(p_j) / densityj2;
      TF3 a = cn<TF>().particle_mass * (dpi + dpj) *
              displacement_cubic_kernel_grad(xixj);
      TF3 force = cn<TF>().particle_mass * a;
      ai -= a;
      particle_force(boundary_id, p_i) += force;
      particle_torque(boundary_id, p_i) +=
          cross(x_j - rigid_x(boundary_id), force);
    }
    particle_pressure_accel(p_i) = cn<TF>().density0 * ai;
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
__global__ void calculate_isph_diagonal_adv_density_with_pellets(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density,
    Variable<1, TF2> particle_diag_adv_density,
    Variable<2, TQ> particle_neighbors, Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary_neighbors,
    Variable<1, U> particle_num_boundary_neighbors, TF dt, U num_particles) {
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
    for (U neighbor_id = 0; neighbor_id < particle_num_boundary_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_boundary_neighbors(p_i, neighbor_id), xixj, p_j);
      TF3 grad_wvol =
          cn<TF>().particle_vol * displacement_cubic_kernel_grad(xixj);
      density_adv += dt * dot(v - particle_v(p_j), grad_wvol);
      diag += (density + cn<TF>().density0) / (density * cn<TF>().density0) /
              (length_sqr(xixj) +
               static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
              dot(xixj, grad_wvol);
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
      pressure =
          max((1 - jacobi_weight) * last_pressure +
                  jacobi_weight * (b / denom - off_diag / diag_adv_density.x),
              static_cast<TF>(0));
    }
    particle_density_err(p_i) =
        pressure == 0 ? 0 : (denom * pressure + off_diag * dt2 - b);
    particle_pressure(p_i) = pressure;
  });
}

template <typename TQ, typename TF3, typename TF2, typename TF>
__global__ void isph_solve_iteration_with_pellets(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_last_pressure, Variable<1, TF> particle_pressure,
    Variable<1, TF2> particle_diag_adv_density,
    Variable<1, TF> particle_density_err, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary_neighbors,
    Variable<1, U> particle_num_boundary_neighbors,

    TF dt, U num_particles) {
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
    for (U neighbor_id = 0; neighbor_id < particle_num_boundary_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_boundary_neighbors(p_i, neighbor_id), xixj, p_j);

      TF densityj = cn<TF>().density0;

      off_diag -= cn<TF>().particle_vol * (density + densityj) /
                  (density * densityj) /
                  (length_sqr(xixj) +
                   static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr) *
                  dot(xixj, displacement_cubic_kernel_grad(xixj)) *
                  particle_last_pressure(p_j);
    }
    off_diag *= cn<TF>().density0;
    if (fabs(denom) > static_cast<TF>(1.0e-9)) {
      pressure =
          max((1 - jacobi_weight) * last_pressure +
                  jacobi_weight * (b / denom - off_diag / diag_adv_density.x),
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
__global__ void compute_dfsph_factor_with_pellets(
    Variable<1, TF3> particle_x, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors,
    Variable<2, TQ> particle_boundary_neighbors,
    Variable<1, U> particle_num_boundary_neighbors,
    Variable<1, TF> particle_dfsph_factor, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF sum_grad_p_k = 0;
    TF3 grad_p_i{0};
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      extract_displacement(particle_neighbors(p_i, neighbor_id), xixj);
      TF3 grad_p_j =
          -cn<TF>().particle_vol * displacement_cubic_kernel_grad(xixj);
      sum_grad_p_k += length_sqr(grad_p_j);
      grad_p_i -= grad_p_j;
    }
    for (U neighbor_id = 0; neighbor_id < particle_num_boundary_neighbors(p_i);
         ++neighbor_id) {
      TF3 grad_wvol;
      extract_displacement(particle_boundary_neighbors(p_i, neighbor_id),
                           grad_wvol);
      TF3 grad_p_j = -grad_wvol;
      sum_grad_p_k += length_sqr(grad_p_j);
      grad_p_i -= grad_p_j;
    }
    sum_grad_p_k += length_sqr(grad_p_i);
    particle_dfsph_factor(p_i) = sum_grad_p_k > cn<TF>().dfsph_factor_epsilon
                                     ? -1 / sum_grad_p_k
                                     : static_cast<TF>(0);
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
      if (fabs(k_sum) > cn<TF>().dfsph_factor_epsilon) {
        TF3 grad_p_j =
            -cn<TF>().particle_vol * displacement_cubic_kernel_grad(xixj);
        dv -= dt * k_sum * grad_p_j;  // ki, kj already contain inverse density
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
    TF3 dv{};  // TODO: add 0?
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
__global__ void pellet_divergence_solve_iteration(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_dx,
    Variable<1, TF> particle_density, Variable<1, TF> particle_dfsph_factor,
    Variable<1, TF> particle_cfl_v2, Variable<2, TQ> particle_neighbors,
    Variable<1, U> particle_num_neighbors, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF b_i = particle_density(p_i) / cn<TF>().density0 - 1;
    TF k_i = b_i * particle_dfsph_factor(p_i);
    TF3 x_i = particle_x(p_i);
    TF3 dx = particle_dx(p_i);
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      const TF b_j = particle_density(p_j) / cn<TF>().density0 - 1;
      const TF kj = b_j * particle_dfsph_factor(p_j);

      const TF kSum = k_i + kj;
      if (fabs(kSum) > cn<TF>().dfsph_factor_epsilon) {
        const TF3 grad_p_j =
            -cn<TF>().particle_vol * displacement_cubic_kernel_grad(xixj);
        dx -= kSum * grad_p_j;
      }
    }
    particle_dx(p_i) = dx;
    particle_cfl_v2(p_i) = length_sqr(dx);
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

// rigid
template <typename TQ, typename TF3, typename TF, typename TDistance>
__global__ void collision_test_with_pellets(
    U j_begin, U j_end, Variable<1, TF3> particle_x,
    Variable<1, U> pellet_id_to_rigid_id,

    Variable<1, U> num_contacts, Variable<1, Contact<TF3, TF>> contacts,
    Variable<1, TF3> x, Variable<1, TF3> v, Variable<1, TQ> q,
    Variable<1, TF3> omega, Variable<1, TF3> mass_contact,
    Variable<1, TF3> inertia,

    const TDistance distance, TF sign,

    U max_num_beads, U num_pellets

) {
  forThreadMappedToElement(num_pellets, [&](U p_i0) {
    U p_i = p_i0 + max_num_beads;
    Contact<TF3, TF> contact;
    U i = pellet_id_to_rigid_id(p_i0);
    contact.cp_i = particle_x(p_i);
    TF3 x_i = x(i);
    for (U j = j_begin; j < j_end; ++j) {
      TF3 x_j = x(j);
      TQ q_i = q(i);
      TQ q_j = q(j);
      TF3 mass_contact_i = mass_contact(i);
      TF3 mass_contact_j = mass_contact(j);

      TF restitution = mass_contact_i.y * mass_contact_j.y;
      TF friction = mass_contact_i.z + mass_contact_j.z;
      TF3 vertex_local_wrt_j = rotate_using_quaternion(
          contact.cp_i - x_j, quaternion_conjugate(q_j));

      TF dist = distance.signed_distance(vertex_local_wrt_j) * sign -
                cn<TF>().contact_tolerance;
      TF3 n =
          distance.gradient(vertex_local_wrt_j, static_cast<TF>(1e-3)) * sign;
      n *= rsqrt(length_sqr(n));  // TODO

      TF3 cp = vertex_local_wrt_j - dist * n;

      if (dist < 0 && i != j &&
          (mass_contact_i.x != 0 || mass_contact_j.x != 0)) {
        U contact_insert_index = atomicAdd(&num_contacts(0), 1);
        contact.i = i;
        contact.j = j;
        contact.cp_j = rotate_using_quaternion(cp, q_j) + x_j;
        contact.n = rotate_using_quaternion(n, q_j);

        contact.friction = friction;

        TF3 r_i = contact.cp_i - x_i;
        TF3 r_j = contact.cp_j - x_j;

        TF3 u_i = v(i) + cross(omega(i), r_i);
        TF3 u_j = v(j) + cross(omega(j), r_j);

        TF3 u_rel = u_i - u_j;
        TF u_rel_n = dot(contact.n, u_rel);

        contact.t = u_rel - u_rel_n * contact.n;
        TF tl2 = length_sqr(contact.t);
        if (tl2 > static_cast<TF>(1e-6)) {
          contact.t = normalize(contact.t);
        }

        calculate_congruent_matrix(1 / inertia(i), q_i, &(contact.iiwi_diag),
                                   &(contact.iiwi_off_diag));
        calculate_congruent_matrix(1 / inertia(j), q_j, &(contact.iiwj_diag),
                                   &(contact.iiwj_off_diag));

        TF3 k_i_diag, k_i_off_diag;
        TF3 k_j_diag, k_j_off_diag;
        calculate_congruent_k(contact.cp_i - x_i, mass_contact_i.x,
                              contact.iiwi_diag, contact.iiwi_off_diag,
                              &k_i_diag, &k_i_off_diag);
        calculate_congruent_k(contact.cp_j - x_j, mass_contact_j.x,
                              contact.iiwj_diag, contact.iiwj_off_diag,
                              &k_j_diag, &k_j_off_diag);
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
        if (contact_insert_index < cni.max_num_contacts) {
          contacts(contact_insert_index) = contact;
        }
      }
    }
  });
}
template <typename TQ, typename TF3, typename TF>
__global__ void collision_test_with_pellets(
    U j_begin, U j_end, Variable<1, TF3> particle_x,
    Variable<1, U> pellet_id_to_rigid_id,

    Variable<1, U> num_contacts, Variable<1, Contact<TF3, TF>> contacts,
    Variable<1, TF3> x, Variable<1, TF3> v, Variable<1, TQ> q,
    Variable<1, TF3> omega, Variable<1, TF3> mass_contact,
    Variable<1, TF3> inertia,

    Variable<1, TF> distance_nodes, TF3 domain_min, TF3 domain_max,
    U3 resolution, TF3 cell_size, TF sign,

    U max_num_beads, U num_pellets

) {
  forThreadMappedToElement(num_pellets, [&](U p_i0) {
    U p_i = p_i0 + max_num_beads;
    Contact<TF3, TF> contact;
    U i = pellet_id_to_rigid_id(p_i0);
    contact.cp_i = particle_x(p_i);
    TF3 x_i = x(i);
    for (U j = j_begin; j < j_end; ++j) {
      TF3 x_j = x(j);
      TQ q_i = q(i);
      TQ q_j = q(j);
      TF3 mass_contact_i = mass_contact(i);
      TF3 mass_contact_j = mass_contact(j);

      TF restitution = mass_contact_i.y * mass_contact_j.y;
      TF friction = mass_contact_i.z + mass_contact_j.z;
      TF3 vertex_local_wrt_j = rotate_using_quaternion(
          contact.cp_i - x_j, quaternion_conjugate(q_j));

      TF3 n;
      TF dist = collision_find_dist_normal(
          &distance_nodes, domain_min, domain_max, resolution, cell_size, sign,
          cn<TF>().contact_tolerance, vertex_local_wrt_j, &n);
      n *= rsqrt(length_sqr(n));  // TODO

      TF3 cp = vertex_local_wrt_j - dist * n;
      if (dist < 0 && i != j &&
          (mass_contact_i.x != 0 || mass_contact_j.x != 0)) {
        U contact_insert_index = atomicAdd(&num_contacts(0), 1);
        contact.i = i;
        contact.j = j;
        contact.cp_j = rotate_using_quaternion(cp, q_j) + x_j;
        contact.n = rotate_using_quaternion(n, q_j);

        contact.friction = friction;

        TF3 r_i = contact.cp_i - x_i;
        TF3 r_j = contact.cp_j - x_j;

        TF3 u_i = v(i) + cross(omega(i), r_i);
        TF3 u_j = v(j) + cross(omega(j), r_j);

        TF3 u_rel = u_i - u_j;
        TF u_rel_n = dot(contact.n, u_rel);

        contact.t = u_rel - u_rel_n * contact.n;
        TF tl2 = length_sqr(contact.t);
        if (tl2 > static_cast<TF>(1e-6)) {
          contact.t = normalize(contact.t);
        }

        calculate_congruent_matrix(1 / inertia(i), q_i, &(contact.iiwi_diag),
                                   &(contact.iiwi_off_diag));
        calculate_congruent_matrix(1 / inertia(j), q_j, &(contact.iiwj_diag),
                                   &(contact.iiwj_off_diag));

        TF3 k_i_diag, k_i_off_diag;
        TF3 k_j_diag, k_j_off_diag;
        calculate_congruent_k(contact.cp_i - x_i, mass_contact_i.x,
                              contact.iiwi_diag, contact.iiwi_off_diag,
                              &k_i_diag, &k_i_off_diag);
        calculate_congruent_k(contact.cp_j - x_j, mass_contact_j.x,
                              contact.iiwj_diag, contact.iiwj_off_diag,
                              &k_j_diag, &k_j_off_diag);
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
        if (contact_insert_index < cni.max_num_contacts) {
          contacts(contact_insert_index) = contact;
        }
      }
    }
  });
}

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
    n *= rsqrt(length_sqr(n));  // TODO

    TF3 cp = vertex_local_wrt_j - dist * n;

    if (dist < 0) {
      U contact_insert_index = atomicAdd(&num_contacts(0), 1);
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
      if (contact_insert_index < cni.max_num_contacts) {
        contacts(contact_insert_index) = contact;
      }
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
    U3 resolution, TF3 cell_size, TF sign,

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
        &distance_nodes, domain_min, domain_max, resolution, cell_size, sign,
        cn<TF>().contact_tolerance, vertex_local_wrt_j, &n);
    TF3 cp = vertex_local_wrt_j - dist * n;

    if (dist < 0) {
      U contact_insert_index = atomicAdd(&num_contacts(0), 1);
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
      if (contact_insert_index < cni.max_num_contacts) {
        contacts(contact_insert_index) = contact;
      }
    }
  });
}

template <U wrap, typename TF3, typename TF, typename TDistance>
__global__ void move_and_avoid_boundary(Variable<1, TF3> particle_x,
                                        Variable<1, TF3> particle_dx,
                                        const TDistance distance, TF sign,
                                        TF cfl, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i) + particle_dx(p_i) * cfl;
    TF dist = distance.signed_distance(x_i) * sign - cn<TF>().contact_tolerance;
    TF3 normal = distance.gradient(x_i, static_cast<TF>(1e-3)) * sign;
    if (dist < cn<TF>().particle_radius) {
      x_i += (cn<TF>().particle_radius - dist) * normalize(normal);
    }
    if constexpr (wrap == 0)
      particle_x(p_i) = x_i;
    else
      particle_x(p_i) = wrap_y(x_i);
  });
}

template <U wrap, typename TF3, typename TF>
__global__ void move_and_avoid_boundary(Variable<1, TF3> particle_x,
                                        Variable<1, TF3> particle_dx,
                                        Variable<1, TF> distance_nodes,
                                        TF3 domain_min, TF3 domain_max,
                                        U3 resolution, TF3 cell_size, TF sign,
                                        TF cfl, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i) + particle_dx(p_i) * cfl;

    // for resolve
    I3 ipos;
    TF3 inner_x;
    TF N[32];
    TF dN0[32];
    TF dN1[32];
    TF dN2[32];
    U cells[32];
    resolve(domain_min, domain_max, resolution, cell_size, x_i, &ipos,
            &inner_x);
    if (ipos.x >= 0) {
      get_shape_function_and_gradient(inner_x, N, dN0, dN1, dN2);
      get_cells(resolution, ipos, cells);
      TF3 normal;
      TF dist = interpolate_and_derive(&distance_nodes, &cell_size, cells, N,
                                       dN0, dN1, dN2, &normal) *
                sign;
      TF nl2 = length_sqr(normal);
      normal *= sign * (nl2 > 0 ? rsqrt(nl2) : 0);
      if (dist < cn<TF>().particle_radius) {
        x_i += (cn<TF>().particle_radius - dist) * normalize(normal);
      }
    }
    if constexpr (wrap == 0)
      particle_x(p_i) = x_i;
    else
      particle_x(p_i) = wrap_y(x_i);
  });
}

template <typename TQ, typename TF3>
__global__ void transform_pellets(Variable<1, TF3> local_pellet_x,
                                  Variable<1, TF3> particle_x,
                                  Variable<1, TF3> particle_v, TF3 rigid_x,
                                  TF3 rigid_v, TQ rigid_q, TF3 rigid_omega,
                                  U num_particles, U offset) {
  forThreadMappedToElement(num_particles, [&](U p_i0) {
    U p_i = p_i0 + offset;
    TF3 local_xi = local_pellet_x(p_i0);
    TF3 rotated_local_x = rotate_using_quaternion(local_xi, rigid_q);
    particle_x(p_i) = rotated_local_x + rigid_x;
    particle_v(p_i) = cross(rigid_omega, rotated_local_x) + rigid_v;
  });
}

// fluid control
//
// template <typename TF3, typename TF>
// __global__ void drive_n_ellipse(
//     Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
//     Variable<1, TF3> particle_a, Variable<2, TF3> focal_x,
//     Variable<2, TF3> focal_v, Variable<1, TF> focal_dist,
//     Variable<1, TF> usher_kernel_radius, Variable<1, TF> drive_strength,
//     U num_ushers, U num_particles) {
//   forThreadMappedToElement(num_particles, [&](U p_i) {
//     const TF3 x_i = particle_x(p_i);
//     const TF3 v_i = particle_v(p_i);
//     TF3 da{0};
//     for (U usher_id = 0; usher_id < num_ushers; ++usher_id) {
//       TF3 fx0 = focal_x(usher_id, 0);
//       TF3 fx1 = focal_x(usher_id, 1);
//       TF3 fx2 = focal_x(usher_id, 2);
//       TF3 fv0 = focal_v(usher_id, 0);
//       TF3 fv1 = focal_v(usher_id, 1);
//       TF3 fv2 = focal_v(usher_id, 2);
//       TF d0 = length(x_i - fx0);
//       TF d1 = length(x_i - fx1);
//       TF d2 = length(x_i - fx2);
//       TF d0d1 = d0 * d1;
//       TF d1d2 = d1 * d2;
//       TF d2d0 = d2 * d0;
//       TF denom = d0d1 + d1d2 + d2d0 +
//                  static_cast<TF>(0.01) * cn<TF>().kernel_radius_sqr;
//       TF3 drive_v = (d0d1 * fv2 + d1d2 * fv0 + d2d0 * fv1) / denom;
//
//       TF uh = usher_kernel_radius(usher_id);
//       da += drive_strength(usher_id) *
//             dist_gaussian_kernel(d0 + d1 + d2 - focal_dist(usher_id), uh) *
//             (drive_v - v_i);
//     }
//     particle_a(p_i) += da;
//   });
// }
template <typename TF3, typename TF>
__global__ void drive_n_ellipse(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF3> particle_guiding, Variable<1, TF3> focal_x,
    Variable<1, TF3> focal_v, Variable<1, TF3> direction,
    Variable<1, TF> usher_kernel_radius, Variable<1, TF> drive_strength,
    U num_ushers, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    const TF3 x_i = particle_x(p_i);
    const TF3 v_i = particle_v(p_i);
    TF3 da{0};
    for (U usher_id = 0; usher_id < num_ushers; ++usher_id) {
      TF3 fx0 = focal_x(usher_id);
      TF3 displacement = x_i - fx0;
      TF d0 = length(displacement);
      TF3 drive_v = focal_v(usher_id);
      TF directional_attenuation =
          dot(displacement / d0, direction(usher_id)) + 1;

      TF uh = usher_kernel_radius(usher_id);
      da += drive_strength(usher_id) * dist_gaussian_kernel(d0, uh) *
            directional_attenuation * (drive_v - v_i);
    }
    particle_guiding(p_i) = da;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void filter_guiding(Variable<1, TF3> particle_guiding,
                               Variable<1, TF3> particle_a,
                               Variable<1, TF> particle_density,
                               Variable<2, TQ> particle_neighbors,
                               Variable<1, U> particle_num_neighbors,
                               U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 da = cn<TF>().cubic_kernel_zero * cn<TF>().particle_mass /
             particle_density(p_i) * particle_guiding(p_i);
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(particle_neighbors(p_i, neighbor_id), xixj, p_j);
      da += displacement_cubic_kernel(xixj) * cn<TF>().particle_mass /
            particle_density(p_j) * particle_guiding(p_j);
    }
    particle_a(p_i) += da;
  });
}

// statistics
template <typename TQ, typename TF3, typename TF, typename TQuantity>
__global__ void sample_fluid(
    Variable<1, TF3> sample_x, Variable<1, TF3> particle_x,
    Variable<1, TF> particle_density, Variable<1, TQuantity> particle_quantity,
    Variable<2, TQ> sample_neighbors, Variable<1, U> sample_num_neighbors,
    Variable<1, TQuantity> sample_quantity, U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TQuantity result{0};
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

template <typename TQ, typename TF>
__global__ void sample_fluid_density(Variable<2, TQ> sample_neighbors,
                                     Variable<1, U> sample_num_neighbors,
                                     Variable<1, TF> sample_density,
                                     U num_samples) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF density = 0;
    for (U neighbor_id = 0; neighbor_id < sample_num_neighbors(p_i);
         ++neighbor_id) {
      TF3 xixj;
      extract_displacement(sample_neighbors(p_i, neighbor_id), xixj);
      density += cn<TF>().particle_vol * displacement_cubic_kernel(xixj);
    }
    sample_density(p_i) = density * cn<TF>().density0;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_sample_velocity(
    Variable<1, TF3> sample_x, Variable<1, TF3> particle_x,
    Variable<1, TF> particle_density, Variable<1, TF3> particle_v,
    Variable<2, TQ> sample_neighbors, Variable<1, U> sample_num_neighbors,
    Variable<1, TF3> sample_v, Variable<2, TQ> sample_boundary,
    Variable<2, TQ> sample_boundary_kernel, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF3 result{0};
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

template <typename TQ, typename TF3, typename TF>
__global__ void compute_sample_velocity_with_pellets(
    Variable<1, TF3> sample_x, Variable<1, TF> particle_density,
    Variable<1, TF3> particle_v, Variable<2, TQ> sample_neighbors,
    Variable<1, U> sample_num_neighbors, Variable<1, TF3> sample_v,
    Variable<2, TQ> sample_pellet_neighbors,
    Variable<1, U> sample_num_pellet_neighbors, U num_samples) {
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
    for (U neighbor_id = 0; neighbor_id < sample_num_pellet_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(sample_pellet_neighbors(p_i, neighbor_id), xixj, p_j);
      result += cn<TF>().particle_vol * displacement_cubic_kernel(xixj) *
                particle_v(p_j);
    }
    sample_v(p_i) = result;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_sample_vorticity(
    Variable<1, TF3> sample_x, Variable<1, TF3> particle_x,
    Variable<1, TF> particle_density, Variable<1, TF3> particle_v,
    Variable<2, TQ> sample_neighbors, Variable<1, U> sample_num_neighbors,
    Variable<1, TF3> sample_v, Variable<1, TF3> sample_vorticity,
    Variable<2, TQ> sample_boundary, Variable<2, TQ> sample_boundary_kernel,
    Variable<1, TF3> rigid_x, Variable<1, TF3> rigid_v,
    Variable<1, TF3> rigid_omega, U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF3 result{0};
    TF3 x_i = sample_x(p_i);
    TF3 v_i = sample_v(p_i);
    for (U neighbor_id = 0; neighbor_id < sample_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(sample_neighbors(p_i, neighbor_id), xixj, p_j);
      result +=
          cn<TF>().particle_mass / particle_density(p_j) *
          cross(particle_v(p_j) - v_i, displacement_cubic_kernel_grad(xixj));
    }
    for (U boundary_id = 0; boundary_id < cni.num_boundaries; ++boundary_id) {
      TQ boundary = sample_boundary(boundary_id, p_i);
      TQ boundary_kernel = sample_boundary_kernel(boundary_id, p_i);
      TF3 const& bx_j = reinterpret_cast<TF3 const&>(boundary);
      TF3 const& grad_wvol = reinterpret_cast<TF3 const&>(boundary_kernel);

      TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
                 rigid_v(boundary_id);
      result += cross(velj - v_i, grad_wvol);
    }
    sample_vorticity(p_i) = result;
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_sample_vorticity_with_pellets(
    Variable<1, TF3> sample_x, Variable<1, TF3> particle_x,
    Variable<1, TF> particle_density, Variable<1, TF3> particle_v,
    Variable<2, TQ> sample_neighbors, Variable<1, U> sample_num_neighbors,
    Variable<1, TF3> sample_v, Variable<1, TF3> sample_vorticity,
    Variable<2, TQ> sample_pellet_neighbors,
    Variable<1, U> sample_num_pellet_neighbors, U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF3 result{0};
    TF3 x_i = sample_x(p_i);
    TF3 v_i = sample_v(p_i);
    for (U neighbor_id = 0; neighbor_id < sample_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(sample_neighbors(p_i, neighbor_id), xixj, p_j);
      result +=
          cn<TF>().particle_mass / particle_density(p_j) *
          cross(particle_v(p_j) - v_i, displacement_cubic_kernel_grad(xixj));
    }
    for (U neighbor_id = 0; neighbor_id < sample_num_pellet_neighbors(p_i);
         ++neighbor_id) {
      U p_j;
      TF3 xixj;
      extract_pid(sample_pellet_neighbors(p_i, neighbor_id), xixj, p_j);
      result +=
          cn<TF>().particle_vol *
          cross(particle_v(p_j) - v_i, displacement_cubic_kernel_grad(xixj));
    }
    sample_vorticity(p_i) = result;
  });
}

template <typename TF>
__global__ void compute_density_mask(Variable<1, TF> sample_density,
                                     Variable<1, U> mask, U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    mask(p_i) = static_cast<U>(sample_density(p_i) > static_cast<TF>(0));
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_boundary_mask(Variable<1, TF> distance_nodes,
                                      TF3 rigid_x, TQ rigid_q, TF3 domain_min,
                                      TF3 domain_max, U3 resolution,
                                      TF3 cell_size, TF sign,
                                      Variable<1, TF3> sample_x,
                                      TF distance_threshold,
                                      Variable<1, TF> mask, U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF d = interpolate_distance_without_intermediates(
               &distance_nodes, domain_min, domain_max, resolution, cell_size,
               rotate_using_quaternion(sample_x(p_i) - rigid_x,
                                       quaternion_conjugate(rigid_q))) *
           sign;
    mask(p_i) *= (d >= distance_threshold);
  });
}

template <typename TQ, typename TF3, typename TF, typename TDistance>
__global__ void compute_boundary_mask(const TDistance distance, TF3 rigid_x,
                                      TQ rigid_q, TF sign,
                                      Variable<1, TF3> sample_x,
                                      TF distance_threshold,
                                      Variable<1, TF> mask, U num_samples) {
  forThreadMappedToElement(num_samples, [&](U p_i) {
    TF d = distance.signed_distance(rotate_using_quaternion(
               sample_x(p_i) - rigid_x, quaternion_conjugate(rigid_q))) *
           sign;
    mask(p_i) *= (d >= distance_threshold);
  });
}

// histogram
inline __device__ void add_byte(U* shared_warp_histogram, U quantized4) {
  atomicAdd(shared_warp_histogram + quantized4, 1);
}
inline __device__ void add_word(U* shared_warp_histogram, U quantized4) {
  add_byte(shared_warp_histogram, quantized4 & 0xFFU);
  add_byte(shared_warp_histogram, (quantized4 >> 8) & 0xFFU);
  add_byte(shared_warp_histogram, (quantized4 >> 16) & 0xFFU);
  add_byte(shared_warp_histogram, (quantized4 >> 24) & 0xFFU);
}
inline __device__ void add_word_with_mask(U* shared_warp_histogram,
                                          U quantized4, U mask4) {
  if (mask4 & 0x01U) {
    add_byte(shared_warp_histogram, quantized4 & 0xFFU);
  }
  if (mask4 & 0x02U) {
    add_byte(shared_warp_histogram, (quantized4 >> 8) & 0xFFU);
  }
  if (mask4 & 0x04U) {
    add_byte(shared_warp_histogram, (quantized4 >> 16) & 0xFFU);
  }
  if (mask4 & 0x08U) {
    add_byte(shared_warp_histogram, (quantized4 >> 24) & 0xFFU);
  }
}
template <typename TU>
__global__ void histogram256(Variable<1, TU> partial_histogram,
                             Variable<1, TU> quantized4s, TU n) {
  cooperative_groups::thread_block cta =
      cooperative_groups::this_thread_block();
  __shared__ TU shared_histogram[kHistogram256ThreadblockMemory];
  TU* shared_warp_histogram =
      shared_histogram + (threadIdx.x >> kLog2WarpSize) * kHistogram256BinCount;
#pragma unroll
  for (TU i = 0;
       i < (kHistogram256ThreadblockMemory / kHistogram256ThreadblockSize);
       ++i) {
    shared_histogram[threadIdx.x + i * kHistogram256ThreadblockSize] = 0;
  }
  cooperative_groups::sync(cta);
  for (TU pos = blockIdx.x * blockDim.x + threadIdx.x; pos < ((n - 1) / 4 + 1);
       pos += blockDim.x * gridDim.x) {
    TU quantized4 = quantized4s(pos);
    add_word(shared_warp_histogram, quantized4);
  }
  cooperative_groups::sync(cta);
  for (TU bin = threadIdx.x; bin < kHistogram256BinCount;
       bin += kHistogram256ThreadblockSize) {
    TU sum = 0;
    for (TU i = 0; i < kWarpCount; ++i) {
      sum += shared_histogram[bin + i * kHistogram256BinCount] & 0xFFFFFFFFU;
    }
    partial_histogram(blockIdx.x * kHistogram256BinCount + bin) = sum;
  }
}

template <typename TU, typename TF>
__global__ void histogram256_with_mask(Variable<1, TU> partial_histogram,
                                       Variable<1, TU> quantized4s,
                                       Variable<1, TF> mask, TU n) {
  cooperative_groups::thread_block cta =
      cooperative_groups::this_thread_block();
  __shared__ TU shared_histogram[kHistogram256ThreadblockMemory];
  TU* shared_warp_histogram =
      shared_histogram + (threadIdx.x >> kLog2WarpSize) * kHistogram256BinCount;
#pragma unroll
  for (TU i = 0;
       i < (kHistogram256ThreadblockMemory / kHistogram256ThreadblockSize);
       ++i) {
    shared_histogram[threadIdx.x + i * kHistogram256ThreadblockSize] = 0;
  }
  cooperative_groups::sync(cta);
  for (TU pos = blockIdx.x * blockDim.x + threadIdx.x; pos < ((n - 1) / 4 + 1);
       pos += blockDim.x * gridDim.x) {
    TU mask4 = 0;
    mask4 += ((mask(pos * 4) == 1) << 3);
    mask4 += ((pos * 4 + 1 < n) ? (mask(pos * 4 + 1) == 1) : 1) << 2;
    mask4 += ((pos * 4 + 2 < n) ? (mask(pos * 4 + 2) == 1) : 1) << 1;
    mask4 += ((pos * 4 + 3 < n) ? (mask(pos * 4 + 3) == 1) : 1);
    TU quantized4 = quantized4s(pos);
    add_word_with_mask(shared_warp_histogram, quantized4, mask4);
  }
  cooperative_groups::sync(cta);
  for (TU bin = threadIdx.x; bin < kHistogram256BinCount;
       bin += kHistogram256ThreadblockSize) {
    TU sum = 0;
    for (TU i = 0; i < kWarpCount; ++i) {
      sum += shared_histogram[bin + i * kHistogram256BinCount] & 0xFFFFFFFFU;
    }
    partial_histogram(blockIdx.x * kHistogram256BinCount + bin) = sum;
  }
}

template <typename TU>
__global__ void merge_histogram256(Variable<1, TU> histogram,
                                   Variable<1, TU> partial_histogram,
                                   TU partial_histogram_size, TU n) {
  cooperative_groups::thread_block cta =
      cooperative_groups::this_thread_block();
  TU sum = 0;
  for (TU i = threadIdx.x; i < partial_histogram_size;
       i += kMergeThreadblockSize) {
    sum += partial_histogram(blockIdx.x + i * kHistogram256BinCount);
  }
  __shared__ TU data[kMergeThreadblockSize];
  data[threadIdx.x] = sum;
  for (TU stride = kMergeThreadblockSize / 2; stride > 0; stride >>= 1) {
    cooperative_groups::sync(cta);
    if (threadIdx.x < stride) {
      data[threadIdx.x] += data[threadIdx.x + stride];
    }
  }
  if (threadIdx.x == 0) {
    histogram(blockIdx.x) =
        data[0] - (blockIdx.x == 0 ? ((4 - (n % 4)) % 4) : 0);
  }
}
template <typename TF>
__global__ void quantize(Variable<1, TF> v, Variable<1, U> quantized4s,
                         TF v_min, TF step_inverse, U n) {
  forThreadMappedToElement((n - 1) / 4 + 1, [&](U row_id) {
    U result = 0;
    U i = row_id * 4;
    result +=
        ((__float2uint_rd((v(i++) - v_min) * step_inverse) & 0xFFU) << 24);
    if (i < n) {
      result +=
          ((__float2uint_rd((v(i++) - v_min) * step_inverse) & 0xFFU) << 16);
    }
    if (i < n) {
      result +=
          ((__float2uint_rd((v(i++) - v_min) * step_inverse) & 0xFFU) << 8);
    }
    if (i < n) {
      result += ((__float2uint_rd((v(i++) - v_min) * step_inverse) & 0xFFU));
    }
    quantized4s(row_id) = result;
  });
}

// Graphical post-processing
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
    if (const char* default_block_size_str =
            std::getenv("AL_DEFAULT_BLOCK_SIZE")) {
      default_block_size_ = std::stoul(default_block_size_str);
    }
    load_optimal_block_size();
  }
  virtual ~Runner() {
    std::stringstream filename_stream;
    filename_stream << ".alcache/";
    if (optimal_block_size_dict_.empty()) {
      filename_stream << default_block_size_;
    } else {
      filename_stream << "optimized";
    }
    filename_stream << ".yaml";
    save_stat(filename_stream.str().c_str());
  }

  template <U D, typename M>
  static void sqrt_inplace(Variable<D, M> var, U num_elements, U offset = 0) {
    thrust::transform(thrust::device_ptr<M>(static_cast<M*>(var.ptr_)) + offset,
                      thrust::device_ptr<M>(static_cast<M*>(var.ptr_)) +
                          (offset + num_elements),
                      thrust::device_ptr<M>(static_cast<M*>(var.ptr_)) + offset,
                      SqrtOperation<M>());
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

  template <U D, typename M>
  static void sort(Variable<D, M> var, U num_elements, U offset = 0) {
    thrust::sort(thrust::device_ptr<M>(static_cast<M*>(var.ptr_)) + offset,
                 thrust::device_ptr<M>(static_cast<M*>(var.ptr_)) +
                     (offset + num_elements));
  }

  template <U D, typename M>
  static U count(Variable<D, M> var, M const& value, U num_elements,
                 U offset = 0) {
    return thrust::count(
        thrust::device_ptr<M>(static_cast<M*>(var.ptr_)) + offset,
        thrust::device_ptr<M>(static_cast<M*>(var.ptr_)) +
            (offset + num_elements),
        value);
  }

  template <U D, typename M>
  static void max_inplace(Variable<D, M>& target, Variable<D, M> const& other,
                          U num_elements, U offset = 0) {
    thrust::transform(
        thrust::device_ptr<M>(static_cast<M*>(other.ptr_)) + offset,
        thrust::device_ptr<M>(static_cast<M*>(other.ptr_)) +
            (offset + num_elements),
        thrust::device_ptr<M>(static_cast<M*>(target.ptr_)) + offset,
        thrust::device_ptr<M>(static_cast<M*>(target.ptr_)) + offset,
        thrust::maximum<M>());
  }
  template <U D, typename M>
  static void max_inplace(Variable<D, M>& target, Variable<D, M> const& other) {
    max_inplace(target, other, target.get_linear_shape());
  }

  template <U D, typename M, typename TPrimitive>
  static TPrimitive calculate_se_weighted(Variable<D, M> v0, Variable<D, M> v1,
                                          Variable<D, TPrimitive> weight0,
                                          U num_elements, U offset = 0) {
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<M>(static_cast<M*>(v0.ptr_)) + offset,
        thrust::device_ptr<M>(static_cast<M*>(v1.ptr_)) + offset,
        thrust::device_ptr<TPrimitive>(static_cast<TPrimitive*>(weight0.ptr_)) +
            offset));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<M>(static_cast<M*>(v0.ptr_)) +
            (offset + num_elements),
        thrust::device_ptr<M>(static_cast<M*>(v1.ptr_)) +
            (offset + num_elements),
        thrust::device_ptr<TPrimitive>(static_cast<TPrimitive*>(weight0.ptr_)) +
            (offset + num_elements)));
    return thrust::transform_reduce(
        begin, end, SquaredDifferenceWeighted<M, TPrimitive>(),
        static_cast<TPrimitive>(0), thrust::plus<TPrimitive>());
  }

  template <U D, typename M, typename TPrimitive>
  static TPrimitive calculate_se_yz_weighted(Variable<D, M> v0,
                                             Variable<D, M> v1,
                                             Variable<D, TPrimitive> mask,
                                             U num_elements, U offset = 0) {
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<M>(static_cast<M*>(v0.ptr_)) + offset,
        thrust::device_ptr<M>(static_cast<M*>(v1.ptr_)) + offset,
        thrust::device_ptr<TPrimitive>(static_cast<TPrimitive*>(mask.ptr_)) +
            offset));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<M>(static_cast<M*>(v0.ptr_)) +
            (offset + num_elements),
        thrust::device_ptr<M>(static_cast<M*>(v1.ptr_)) +
            (offset + num_elements),
        thrust::device_ptr<TPrimitive>(static_cast<TPrimitive*>(mask.ptr_)) +
            (offset + num_elements)));
    return thrust::transform_reduce(
        begin, end, SquaredDifferenceYzWeighted<M, TPrimitive>(),
        static_cast<TPrimitive>(0), thrust::plus<TPrimitive>());
  }

  static TF calculate_kl_divergence(Variable<1, U> histogram_p,
                                    Variable<1, U> histogram_q, U n_p, U n_q,
                                    TF q_lower_bound, U num_bins) {
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<U>(static_cast<U*>(histogram_p.ptr_)),
        thrust::device_ptr<U>(static_cast<U*>(histogram_q.ptr_))));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<U>(static_cast<U*>(histogram_p.ptr_)) + (num_bins),
        thrust::device_ptr<U>(static_cast<U*>(histogram_q.ptr_)) + (num_bins)));
    return thrust::transform_reduce(begin, end,
                                    KLDivergence(n_p, n_q, q_lower_bound),
                                    static_cast<TF>(0), thrust::plus<TF>());
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
    float elapsed;
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
        cudaEventElapsedTime(&elapsed, event_start, event_stop));
    std::pair<U, float>& count_and_mean =
        launch_stat_dict_[name][desired_block_size / kWarpSize - 1];
    ++count_and_mean.first;
    count_and_mean.second +=
        (elapsed - count_and_mean.second) / count_and_mean.first;
    Allocator::abort_if_error(cudaEventDestroy(event_start));
    Allocator::abort_if_error(cudaEventDestroy(event_stop));
  }

  void record_elapsed_start(std::string name) {
    cudaEvent_t& event_start = custom_event_dict_[name];
    Allocator::abort_if_error(cudaEventCreate(&event_start));
    Allocator::abort_if_error(cudaGetLastError());
    Allocator::abort_if_error(cudaEventRecord(event_start));
  }

  void record_elapsed_end(std::string name) {
    cudaEvent_t& event_start = custom_event_dict_[name];
    cudaEvent_t event_stop;
    float elapsed;
    Allocator::abort_if_error(cudaEventCreate(&event_stop));
    Allocator::abort_if_error(cudaGetLastError());
    Allocator::abort_if_error(cudaEventRecord(event_stop));
    Allocator::abort_if_error(cudaEventSynchronize(event_stop));
    Allocator::abort_if_error(
        cudaEventElapsedTime(&elapsed, event_start, event_stop));
    custom_elapsed_dict_[name] = elapsed;
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
      } else {
        block_size = optimal_block_size_dict_[name];
      }
    }
    if (block_size >
        attr.maxThreadsPerBlock) {  // max. block size can be smaller than 1024
                                    // depending on the kernel
      block_size = attr.maxThreadsPerBlock;
    }
    launch(n, block_size, f, name);
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
      for (auto const& count_and_mean : item.second) {
        stream << std::setprecision(std::numeric_limits<float>::max_digits10)
               << (count_and_mean.first == 0 ? -1.0f : count_and_mean.second)
               << ", ";
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
  void launch_print_cn() {
    launch(
        1,
        [&](U grid_size, U block_size) {
          print_cn<TF><<<grid_size, block_size>>>();
        },
        "print_cn", print_cn<TF>);
  }
  void launch_create_fluid_block(Variable<1, TF3>& particle_x, U num_particles,
                                 U offset, TF particle_radius, int mode,
                                 TF3 const& box_min, TF3 const& box_max) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          create_fluid_block<TF3, TF><<<grid_size, block_size>>>(
              particle_x, num_particles, offset, particle_radius, mode, box_min,
              box_max);
        },
        "create_fluid_block", create_fluid_block<TF3, TF>);
  }
  void launch_create_custom_beads_internal(
      Variable<1, TF3>& particle_x, Variable<1, TF3> const& ref_x,
      Variable<1, U>& internal_encoded_sorted, U num_particles, U offset) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          create_custom_beads_internal<TF3, TF><<<grid_size, block_size>>>(
              particle_x, ref_x, internal_encoded_sorted, num_particles,
              offset);
        },
        "create_custom_beads_internal", create_custom_beads_internal<TF3, TF>);
  }
  void launch_create_custom_beads_scalar_internal(
      Variable<1, TF>& particle_scalar, Variable<1, TF> const& ref_scalar,
      Variable<1, U>& internal_encoded_sorted, U num_particles, U offset) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          create_custom_beads_scalar_internal<TF><<<grid_size, block_size>>>(
              particle_scalar, ref_scalar, internal_encoded_sorted,
              num_particles, offset);
        },
        "create_custom_beads_scalar_internal",
        create_custom_beads_scalar_internal<TF>);
  }
  void launch_create_fluid_block_internal(
      Variable<1, TF3>& particle_x, Variable<1, U>& internal_encoded_sorted,
      U num_particles, U offset, TF particle_radius, int mode,
      TF3 const& box_min, TF3 const& box_max) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          create_fluid_block_internal<TF3, TF><<<grid_size, block_size>>>(
              particle_x, internal_encoded_sorted, num_particles, offset,
              particle_radius, mode, box_min, box_max);
        },
        "create_fluid_block_internal", create_fluid_block_internal<TF3, TF>);
  }
  void launch_create_fluid_cylinder_internal(
      Variable<1, TF3>& particle_x, Variable<1, U>& internal_encoded_sorted,
      U num_particles, U offset, TF radius, TF particle_radius, TF y_min,
      TF y_max) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          create_fluid_cylinder_internal<TF3, TF><<<grid_size, block_size>>>(
              particle_x, internal_encoded_sorted, num_particles, offset,
              radius, particle_radius, y_min, y_max);
        },
        "create_fluid_cylinder_internal",
        create_fluid_cylinder_internal<TF3, TF>);
  }
  void launch_compute_custom_beads_internal(
      Variable<1, U>& internal_encoded, Variable<1, TF3> const& bead_x,
      dg::Distance<TF3, TF> const& virtual_dist,
      Variable<1, TF> const& distance_nodes, TF3 const& domain_min,
      TF3 const& domain_max, U3 const& resolution, TF3 const& cell_size,
      TF sign, TF3 const& rigid_x, TQ const& rigid_q, U num_positions) {
    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TBoxShellDistance = dg::BoxShellDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    using TInfiniteTubeDistance = dg::InfiniteTubeDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_custom_beads_internal<<<grid_size, block_size>>>(
                internal_encoded, bead_x, distance_nodes, domain_min,
                domain_max, resolution, cell_size, sign, rigid_x, rigid_q,
                num_positions);
          },
          "compute_custom_beads_internal",
          compute_custom_beads_internal<TQ, TF3, TF>);
    } else if (TBoxDistance const* distance =
                   dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_custom_beads_internal<<<grid_size, block_size>>>(
                internal_encoded, bead_x, *distance, sign, rigid_x, rigid_q,
                num_positions);
          },
          "compute_custom_beads_internal(BoxDistance)",
          compute_custom_beads_internal<TQ, TF3, TF, TBoxDistance>);
    } else if (TBoxShellDistance const* distance =
                   dynamic_cast<TBoxShellDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_custom_beads_internal<<<grid_size, block_size>>>(
                internal_encoded, bead_x, *distance, sign, rigid_x, rigid_q,
                num_positions);
          },
          "compute_custom_beads_internal(BoxShellDistance)",
          compute_custom_beads_internal<TQ, TF3, TF, TBoxShellDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_custom_beads_internal<<<grid_size, block_size>>>(
                internal_encoded, bead_x, *distance, sign, rigid_x, rigid_q,
                num_positions);
          },
          "compute_custom_beads_internal(SphereDistance)",
          compute_custom_beads_internal<TQ, TF3, TF, TSphereDistance>);
    } else if (TCylinderDistance const* distance =
                   dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_custom_beads_internal<<<grid_size, block_size>>>(
                internal_encoded, bead_x, *distance, sign, rigid_x, rigid_q,
                num_positions);
          },
          "compute_custom_beads_internal(CylinderDistance)",
          compute_custom_beads_internal<TQ, TF3, TF, TCylinderDistance>);
    } else if (TInfiniteCylinderDistance const* distance =
                   dynamic_cast<TInfiniteCylinderDistance const*>(
                       &virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_custom_beads_internal<<<grid_size, block_size>>>(
                internal_encoded, bead_x, *distance, sign, rigid_x, rigid_q,
                num_positions);
          },
          "compute_custom_beads_internal(InfiniteCylinderDistance)",
          compute_custom_beads_internal<TQ, TF3, TF,
                                        TInfiniteCylinderDistance>);
    } else if (TInfiniteTubeDistance const* distance =
                   dynamic_cast<TInfiniteTubeDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_custom_beads_internal<<<grid_size, block_size>>>(
                internal_encoded, bead_x, *distance, sign, rigid_x, rigid_q,
                num_positions);
          },
          "compute_custom_beads_internal(InfiniteTubeDistance)",
          compute_custom_beads_internal<TQ, TF3, TF, TInfiniteTubeDistance>);
    } else if (TCapsuleDistance const* distance =
                   dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_custom_beads_internal<<<grid_size, block_size>>>(
                internal_encoded, bead_x, *distance, sign, rigid_x, rigid_q,
                num_positions);
          },
          "compute_custom_beads_internal(CapsuleDistance)",
          compute_custom_beads_internal<TQ, TF3, TF, TCapsuleDistance>);
    } else {
      std::cerr
          << "[compute_custom_beads_internal] Distance type not supported."
          << std::endl;
    }
  }
  void launch_compute_fluid_block_internal(
      Variable<1, U>& internal_encoded,
      dg::Distance<TF3, TF> const& virtual_dist,
      Variable<1, TF> const& distance_nodes, TF3 const& domain_min,
      TF3 const& domain_max, U3 const& resolution, TF3 const& cell_size,
      TF sign, TF3 const& rigid_x, TQ const& rigid_q, U num_positions,
      TF particle_radius, int mode, TF3 const& box_min, TF3 const& box_max) {
    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TBoxShellDistance = dg::BoxShellDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    using TInfiniteTubeDistance = dg::InfiniteTubeDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_block_internal<<<grid_size, block_size>>>(
                internal_encoded, distance_nodes, domain_min, domain_max,
                resolution, cell_size, sign, rigid_x, rigid_q, num_positions,
                particle_radius, mode, box_min, box_max);
          },
          "compute_fluid_block_internal",
          compute_fluid_block_internal<TQ, TF3, TF>);
    } else if (TBoxDistance const* distance =
                   dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_block_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, particle_radius, mode, box_min, box_max);
          },
          "compute_fluid_block_internal(BoxDistance)",
          compute_fluid_block_internal<TQ, TF3, TF, TBoxDistance>);
    } else if (TBoxShellDistance const* distance =
                   dynamic_cast<TBoxShellDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_block_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, particle_radius, mode, box_min, box_max);
          },
          "compute_fluid_block_internal(BoxShellDistance)",
          compute_fluid_block_internal<TQ, TF3, TF, TBoxShellDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_block_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, particle_radius, mode, box_min, box_max);
          },
          "compute_fluid_block_internal(SphereDistance)",
          compute_fluid_block_internal<TQ, TF3, TF, TSphereDistance>);
    } else if (TCylinderDistance const* distance =
                   dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_block_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, particle_radius, mode, box_min, box_max);
          },
          "compute_fluid_block_internal(CylinderDistance)",
          compute_fluid_block_internal<TQ, TF3, TF, TCylinderDistance>);
    } else if (TInfiniteCylinderDistance const* distance =
                   dynamic_cast<TInfiniteCylinderDistance const*>(
                       &virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_block_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, particle_radius, mode, box_min, box_max);
          },
          "compute_fluid_block_internal(InfiniteCylinderDistance)",
          compute_fluid_block_internal<TQ, TF3, TF, TInfiniteCylinderDistance>);
    } else if (TInfiniteTubeDistance const* distance =
                   dynamic_cast<TInfiniteTubeDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_block_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, particle_radius, mode, box_min, box_max);
          },
          "compute_fluid_block_internal(InfiniteTubeDistance)",
          compute_fluid_block_internal<TQ, TF3, TF, TInfiniteTubeDistance>);
    } else if (TCapsuleDistance const* distance =
                   dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_block_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, particle_radius, mode, box_min, box_max);
          },
          "compute_fluid_block_internal(CapsuleDistance)",
          compute_fluid_block_internal<TQ, TF3, TF, TCapsuleDistance>);
    } else {
      std::cerr << "[compute_fluid_block_internal] Distance type not supported."
                << std::endl;
    }
  }
  void launch_compute_fluid_cylinder_internal(
      Variable<1, U>& internal_encoded,
      dg::Distance<TF3, TF> const& virtual_dist,
      Variable<1, TF> const& distance_nodes, TF3 const& domain_min,
      TF3 const& domain_max, U3 const& resolution, TF3 const& cell_size,
      TF sign, TF3 const& rigid_x, TQ const& rigid_q, U num_positions,
      TF radius, TF particle_radius, TF y_min, TF y_max) {
    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TBoxShellDistance = dg::BoxShellDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    using TInfiniteTubeDistance = dg::InfiniteTubeDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_cylinder_internal<<<grid_size, block_size>>>(
                internal_encoded, distance_nodes, domain_min, domain_max,
                resolution, cell_size, sign, rigid_x, rigid_q, num_positions,
                radius, particle_radius, y_min, y_max);
          },
          "compute_fluid_cylinder_internal",
          compute_fluid_cylinder_internal<TQ, TF3, TF>);
    } else if (TBoxDistance const* distance =
                   dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_cylinder_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, radius, particle_radius, y_min, y_max);
          },
          "compute_fluid_cylinder_internal(BoxDistance)",
          compute_fluid_cylinder_internal<TQ, TF3, TF, TBoxDistance>);
    } else if (TBoxShellDistance const* distance =
                   dynamic_cast<TBoxShellDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_cylinder_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, radius, particle_radius, y_min, y_max);
          },
          "compute_fluid_cylinder_internal(BoxShellDistance)",
          compute_fluid_cylinder_internal<TQ, TF3, TF, TBoxShellDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_cylinder_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, radius, particle_radius, y_min, y_max);
          },
          "compute_fluid_cylinder_internal(SphereDistance)",
          compute_fluid_cylinder_internal<TQ, TF3, TF, TSphereDistance>);
    } else if (TCylinderDistance const* distance =
                   dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_cylinder_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, radius, particle_radius, y_min, y_max);
          },
          "compute_fluid_cylinder_internal(CylinderDistance)",
          compute_fluid_cylinder_internal<TQ, TF3, TF, TCylinderDistance>);
    } else if (TInfiniteCylinderDistance const* distance =
                   dynamic_cast<TInfiniteCylinderDistance const*>(
                       &virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_cylinder_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, radius, particle_radius, y_min, y_max);
          },
          "compute_fluid_cylinder_internal(InfiniteCylinderDistance)",
          compute_fluid_cylinder_internal<TQ, TF3, TF,
                                          TInfiniteCylinderDistance>);
    } else if (TInfiniteTubeDistance const* distance =
                   dynamic_cast<TInfiniteTubeDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_cylinder_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, radius, particle_radius, y_min, y_max);
          },
          "compute_fluid_cylinder_internal(InfiniteTubeDistance)",
          compute_fluid_cylinder_internal<TQ, TF3, TF, TInfiniteTubeDistance>);
    } else if (TCapsuleDistance const* distance =
                   dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
      launch(
          num_positions,
          [&](U grid_size, U block_size) {
            compute_fluid_cylinder_internal<<<grid_size, block_size>>>(
                internal_encoded, *distance, sign, rigid_x, rigid_q,
                num_positions, radius, particle_radius, y_min, y_max);
          },
          "compute_fluid_cylinder_internal(CapsuleDistance)",
          compute_fluid_cylinder_internal<TQ, TF3, TF, TCapsuleDistance>);
    } else {
      std::cerr
          << "[compute_fluid_cylinder_internal] Distance type not supported."
          << std::endl;
    }
  }
  static U get_fluid_block_num_particles(int mode, TF3 box_min, TF3 box_max,
                                         TF particle_radius) {
    I3 steps;
    get_fluid_block_attr(mode, box_min, box_max, particle_radius, steps);
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
  void launch_create_fluid_cylinder(Variable<1, TF3>& particle_x,
                                    U num_particles, U offset, TF radius,
                                    TF particle_radius, TF y_min, TF y_max) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          create_fluid_cylinder<TF3, TF><<<grid_size, block_size>>>(
              particle_x, num_particles, offset, radius, particle_radius, y_min,
              y_max);
        },
        "create_fluid_cylinder", create_fluid_cylinder<TF3, TF>);
  }
  void launch_compute_particle_boundary(
      dg::Distance<TF3, TF> const& virtual_dist,
      Variable<1, TF> const& volume_nodes,
      Variable<1, TF> const& distance_nodes, TF3 const& rigid_x,
      TQ const& rigid_q, U boundary_id, TF3 const& domain_min,
      TF3 const& domain_max, U3 const& resolution, TF3 const& cell_size,
      TF sign, Variable<1, TF3>& particle_x, Variable<2, TQ>& particle_boundary,
      Variable<2, TQ>& particle_boundary_kernel, U num_particles) {
    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TBoxShellDistance = dg::BoxShellDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    using TInfiniteTubeDistance = dg::InfiniteTubeDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary<<<grid_size, block_size>>>(
                volume_nodes, distance_nodes, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, sign, particle_x,
                particle_boundary, particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary", compute_particle_boundary<TQ, TF3, TF>);
    } else if (TBoxDistance const* distance =
                   dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<0><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, sign, particle_x,
                particle_boundary, particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(BoxDistance)",
          compute_particle_boundary_analytic<0, TQ, TF3, TF, TBoxDistance>);
    } else if (TBoxShellDistance const* distance =
                   dynamic_cast<TBoxShellDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<0><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, sign, particle_x,
                particle_boundary, particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(BoxShellDistance)",
          compute_particle_boundary_analytic<0, TQ, TF3, TF,
                                             TBoxShellDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<0><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, sign, particle_x,
                particle_boundary, particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(SphereDistance)",
          compute_particle_boundary_analytic<0, TQ, TF3, TF, TSphereDistance>);
    } else if (TCylinderDistance const* distance =
                   dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<0><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, sign, particle_x,
                particle_boundary, particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(CylinderDistance)",
          compute_particle_boundary_analytic<0, TQ, TF3, TF,
                                             TCylinderDistance>);
    } else if (TInfiniteCylinderDistance const* distance =
                   dynamic_cast<TInfiniteCylinderDistance const*>(
                       &virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<1><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, sign, particle_x,
                particle_boundary, particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(InfiniteCylinderDistance)",
          compute_particle_boundary_analytic<1, TQ, TF3, TF,
                                             TInfiniteCylinderDistance>);
    } else if (TInfiniteTubeDistance const* distance =
                   dynamic_cast<TInfiniteTubeDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<1><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, sign, particle_x,
                particle_boundary, particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(InfiniteTubeDistance)",
          compute_particle_boundary_analytic<1, TQ, TF3, TF,
                                             TInfiniteTubeDistance>);
    } else if (TCapsuleDistance const* distance =
                   dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_particle_boundary_analytic<0><<<grid_size, block_size>>>(
                volume_nodes, *distance, rigid_x, rigid_q, boundary_id,
                domain_min, domain_max, resolution, cell_size, sign, particle_x,
                particle_boundary, particle_boundary_kernel, num_particles);
          },
          "compute_particle_boundary_analytic(CapsuleDistance)",
          compute_particle_boundary_analytic<0, TQ, TF3, TF, TCapsuleDistance>);
    } else {
      std::cerr << "[compute_particle_boundary] Distance type not supported."
                << std::endl;
    }
  }
  void launch_compute_particle_boundary_with_pellets(
      Variable<1, TQ>& particle_boundary_kernel_combined,
      Variable<2, TQ>& sample_pellet_neighbors,
      Variable<1, U>& sample_num_pellet_neighbors, U num_particles) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_particle_boundary_with_pellets<<<grid_size, block_size>>>(
              particle_boundary_kernel_combined, sample_pellet_neighbors,
              sample_num_pellet_neighbors, num_particles);
        },
        "compute_particle_boundary_with_pellets",
        compute_particle_boundary_with_pellets<TQ>);
  }
  void launch_compute_density_mask(Variable<1, TF> const& sample_density,
                                   Variable<1, U>& mask, U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          compute_density_mask<<<grid_size, block_size>>>(sample_density, mask,
                                                          num_samples);
        },
        "compute_density_mask", compute_density_mask<TF>);
  }
  void launch_compute_boundary_mask(dg::Distance<TF3, TF> const& virtual_dist,
                                    Variable<1, TF> const& distance_nodes,
                                    TF3 const& rigid_x, TQ const& rigid_q,
                                    TF3 const& domain_min,
                                    TF3 const& domain_max, U3 const& resolution,
                                    TF3 const& cell_size, TF sign,
                                    Variable<1, TF3> const& sample_x,
                                    TF distance_threshold,
                                    Variable<1, TF>& mask, U num_samples) {
    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    // NOTE: InfiniteCylinderDistance and InfiniteTubeDistance omitted
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      launch(
          num_samples,
          [&](U grid_size, U block_size) {
            compute_boundary_mask<<<grid_size, block_size>>>(
                distance_nodes, rigid_x, rigid_q, domain_min, domain_max,
                resolution, cell_size, sign, sample_x, distance_threshold, mask,
                num_samples);
          },
          "compute_boundary_mask", compute_boundary_mask<TQ, TF3, TF>);
    } else if (TBoxDistance const* distance =
                   dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_samples,
          [&](U grid_size, U block_size) {
            compute_boundary_mask<<<grid_size, block_size>>>(
                *distance, rigid_x, rigid_q, sign, sample_x, distance_threshold,
                mask, num_samples);
          },
          "compute_boundary_mask(BoxDistance)",
          compute_boundary_mask<TQ, TF3, TF, TBoxDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_samples,
          [&](U grid_size, U block_size) {
            compute_boundary_mask<<<grid_size, block_size>>>(
                *distance, rigid_x, rigid_q, sign, sample_x, distance_threshold,
                mask, num_samples);
          },
          "compute_boundary_mask(SphereDistance)",
          compute_boundary_mask<TQ, TF3, TF, TSphereDistance>);
    } else if (TCylinderDistance const* distance =
                   dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
      launch(
          num_samples,
          [&](U grid_size, U block_size) {
            compute_boundary_mask<<<grid_size, block_size>>>(
                *distance, rigid_x, rigid_q, sign, sample_x, distance_threshold,
                mask, num_samples);
          },
          "compute_boundary_mask(CylinderDistance)",
          compute_boundary_mask<TQ, TF3, TF, TCylinderDistance>);
    } else if (TCapsuleDistance const* distance =
                   dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
      launch(
          num_samples,
          [&](U grid_size, U block_size) {
            compute_boundary_mask<<<grid_size, block_size>>>(
                *distance, rigid_x, rigid_q, sign, sample_x, distance_threshold,
                mask, num_samples);
          },
          "compute_boundary_mask(CapsuleDistance)",
          compute_boundary_mask<TQ, TF3, TF, TCapsuleDistance>);
    } else {
      std::cerr << "[compute_particle_boundary] Distance type not supported."
                << std::endl;
    }
  }
  void launch_update_particle_grid(Variable<1, TF3>& particle_x,
                                   Variable<4, TQ>& pid,
                                   Variable<3, U>& pid_length,
                                   Variable<1, U>& grid_anomaly,
                                   U num_particles, U offset = 0) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          update_particle_grid<<<grid_size, block_size>>>(
              particle_x, pid, pid_length, grid_anomaly, num_particles, offset);
        },
        "update_particle_grid", update_particle_grid<TQ, TF3>);
  }
  template <U wrap>
  void launch_make_neighbor_list(Variable<1, TF3>& sample_x,
                                 Variable<4, TQ>& pid,
                                 Variable<3, U>& pid_length,
                                 Variable<2, TQ>& sample_neighbors,
                                 Variable<1, U>& sample_num_neighbors,
                                 Variable<1, U>& grid_anomaly, U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          make_neighbor_list<wrap><<<grid_size, block_size>>>(
              sample_x, pid, pid_length, sample_neighbors, sample_num_neighbors,
              grid_anomaly, num_samples);
        },
        "make_neighbor_list", make_neighbor_list<wrap, TQ, TF3>);
  }

  template <U wrap>
  void launch_make_bead_pellet_neighbor_list(
      Variable<1, TF3>& sample_x, Variable<4, TQ>& pid,
      Variable<3, U>& pid_length, Variable<2, TQ>& sample_bead_neighbors,
      Variable<1, U>& sample_num_bead_neighbors,
      Variable<2, TQ>& sample_pellet_neighbors,
      Variable<1, U>& sample_num_pellet_neighbors, Variable<1, U>& grid_anomaly,
      U max_num_beads, U num_samples, U offset = 0) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          make_bead_pellet_neighbor_list<wrap><<<grid_size, block_size>>>(
              sample_x, pid, pid_length, sample_bead_neighbors,
              sample_num_bead_neighbors, sample_pellet_neighbors,
              sample_num_pellet_neighbors, grid_anomaly, max_num_beads,
              num_samples, offset);
        },
        "make_bead_pellet_neighbor_list",
        make_bead_pellet_neighbor_list<wrap, TQ, TF3>);
  }
  template <U wrap>
  void launch_make_bead_pellet_neighbor_list_check_contiguous(
      Variable<1, TF3>& sample_x, Variable<4, TQ>& pid,
      Variable<3, U>& pid_length, Variable<2, TQ>& sample_bead_neighbors,
      Variable<1, U>& sample_num_bead_neighbors,
      Variable<2, TQ>& sample_pellet_neighbors,
      Variable<1, U>& sample_num_pellet_neighbors, Variable<1, U>& grid_anomaly,
      U max_num_beads, U num_beads, U num_pellets) {
    if (num_beads == max_num_beads) {
      launch_make_bead_pellet_neighbor_list<wrap>(
          sample_x, pid, pid_length, sample_bead_neighbors,
          sample_num_bead_neighbors, sample_pellet_neighbors,
          sample_num_pellet_neighbors, grid_anomaly, max_num_beads,
          num_beads + num_pellets);
    } else {
      launch_make_bead_pellet_neighbor_list<wrap>(
          sample_x, pid, pid_length, sample_bead_neighbors,
          sample_num_bead_neighbors, sample_pellet_neighbors,
          sample_num_pellet_neighbors, grid_anomaly, max_num_beads, num_beads);
      launch_make_bead_pellet_neighbor_list<wrap>(
          sample_x, pid, pid_length, sample_bead_neighbors,
          sample_num_bead_neighbors, sample_pellet_neighbors,
          sample_num_pellet_neighbors, grid_anomaly, max_num_beads, num_pellets,
          max_num_beads);
    }
  }
  void launch_compute_density(Variable<2, TQ>& particle_neighbors,
                              Variable<1, U>& particle_num_neighbors,
                              Variable<1, TF>& particle_density,
                              Variable<2, TQ>& particle_boundary_kernel,
                              U num_particles) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_density<<<grid_size, block_size>>>(
              particle_neighbors, particle_num_neighbors, particle_density,
              particle_boundary_kernel, num_particles);
        },
        "compute_density", compute_density<TQ, TF>);
  }
  void launch_compute_density_with_pellets(
      Variable<2, TQ>& particle_neighbors,
      Variable<1, U>& particle_num_neighbors,
      Variable<2, TQ>& particle_boundary_neighbors,
      Variable<1, U>& particle_num_boundary_neighbors,
      Variable<1, TF>& particle_density, U num_particles) {
    launch(
        num_particles,
        [&](U grid_size, U block_size) {
          compute_density_with_pellets<<<grid_size, block_size>>>(
              particle_neighbors, particle_num_neighbors,
              particle_boundary_neighbors, particle_num_boundary_neighbors,
              particle_density, num_particles);
        },
        "compute_density_with_pellets", compute_density_with_pellets<TQ, TF>);
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
  void launch_sample_fluid_density(Variable<2, TQ>& sample_neighbors,
                                   Variable<1, U>& sample_num_neighbors,
                                   Variable<1, TF>& sample_density,
                                   U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          sample_fluid_density<<<grid_size, block_size>>>(
              sample_neighbors, sample_num_neighbors, sample_density,
              num_samples);
        },
        "sample_fluid_density", sample_fluid_density<TQ, TF>);
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
          compute_sample_velocity<<<grid_size, block_size>>>(
              sample_x, particle_x, particle_density, particle_v,
              sample_neighbors, sample_num_neighbors, sample_v, sample_boundary,
              sample_boundary_kernel, rigid_x, rigid_v, rigid_omega,
              num_samples);
        },
        "compute_sample_velocity", compute_sample_velocity<TQ, TF3, TF>);
  }
  void launch_sample_velocity_with_pellets(
      Variable<1, TF3>& sample_x, Variable<1, TF3>& particle_x,
      Variable<1, TF>& particle_density, Variable<1, TF3>& particle_v,
      Variable<2, TQ>& sample_neighbors, Variable<1, U>& sample_num_neighbors,
      Variable<1, TF3>& sample_v, Variable<2, TQ>& sample_pellet_neighbors,
      Variable<1, U>& sample_num_pellet_neighbors, U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          compute_sample_velocity_with_pellets<<<grid_size, block_size>>>(
              sample_x, particle_density, particle_v, sample_neighbors,
              sample_num_neighbors, sample_v, sample_pellet_neighbors,
              sample_num_pellet_neighbors, num_samples);
        },
        "compute_sample_velocity_with_pellets",
        compute_sample_velocity_with_pellets<TQ, TF3, TF>);
  }
  void launch_sample_vorticity(
      Variable<1, TF3>& sample_x, Variable<1, TF3>& particle_x,
      Variable<1, TF>& particle_density, Variable<1, TF3>& particle_v,
      Variable<2, TQ>& sample_neighbors, Variable<1, U>& sample_num_neighbors,
      Variable<1, TF3>& sample_v, Variable<1, TF3>& sample_vorticity,
      Variable<2, TQ>& sample_boundary, Variable<2, TQ>& sample_boundary_kernel,
      Variable<1, TF3> rigid_x, Variable<1, TF3> rigid_v,
      Variable<1, TF3> rigid_omega, U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          compute_sample_vorticity<<<grid_size, block_size>>>(
              sample_x, particle_x, particle_density, particle_v,
              sample_neighbors, sample_num_neighbors, sample_v,
              sample_vorticity, sample_boundary, sample_boundary_kernel,
              rigid_x, rigid_v, rigid_omega, num_samples);
        },
        "compute_sample_vorticity", compute_sample_vorticity<TQ, TF3, TF>);
  }
  void launch_sample_vorticity_with_pellets(
      Variable<1, TF3>& sample_x, Variable<1, TF3>& particle_x,
      Variable<1, TF>& particle_density, Variable<1, TF3>& particle_v,
      Variable<2, TQ>& sample_neighbors, Variable<1, U>& sample_num_neighbors,
      Variable<1, TF3>& sample_v, Variable<1, TF3>& sample_vorticity,
      Variable<2, TQ>& sample_pellet_neighbors,
      Variable<1, U>& sample_num_pellet_neighbors, U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          compute_sample_vorticity_with_pellets<<<grid_size, block_size>>>(
              sample_x, particle_x, particle_density, particle_v,
              sample_neighbors, sample_num_neighbors, sample_v,
              sample_vorticity, sample_pellet_neighbors,
              sample_num_pellet_neighbors, num_samples);
        },
        "compute_sample_vorticity_with_pellets",
        compute_sample_vorticity_with_pellets<TQ, TF3, TF>);
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
  void launch_sample_density_with_pellets(
      Variable<1, TF3>& sample_x, Variable<2, TQ>& sample_neighbors,
      Variable<1, U>& sample_num_neighbors,
      Variable<2, TQ>& sample_pellet_neighbors,
      Variable<1, U>& sample_num_pellet_neighbors,
      Variable<1, TF>& sample_density, U num_samples) {
    launch(
        num_samples,
        [&](U grid_size, U block_size) {
          sample_position_density_with_pellets<<<grid_size, block_size>>>(
              sample_x, sample_neighbors, sample_num_neighbors,
              sample_pellet_neighbors, sample_num_pellet_neighbors,
              sample_density, num_samples);
        },
        "sample_position_density_with_pellets",
        sample_position_density_with_pellets<TQ, TF3, TF>);
  }
  void launch_histogram256(Variable<1, U> partial_histogram,
                           Variable<1, U> histogram, Variable<1, U> quantized4s,
                           Variable<1, TF> const& v, TF v_min, TF v_max, U n) {
    TF step_inverse = static_cast<TF>(
        256.0 / (static_cast<double>(v_max) - static_cast<double>(v_min)));
    launch(
        n,
        [&](U grid_size, U block_size) {
          quantize<<<grid_size, block_size>>>(v, quantized4s, v_min,
                                              step_inverse, n);
        },
        "quantize", quantize<TF>);
    histogram256<<<kPartialHistogram256Count, kHistogram256ThreadblockSize>>>(
        partial_histogram, quantized4s, n);
    Allocator::abort_if_error(cudaGetLastError());
    merge_histogram256<<<kHistogram256BinCount, kMergeThreadblockSize>>>(
        histogram, partial_histogram, kPartialHistogram256Count, n);
    Allocator::abort_if_error(cudaGetLastError());
  }
  void launch_histogram256_with_mask(Variable<1, U> partial_histogram,
                                     Variable<1, U> histogram,
                                     Variable<1, U> quantized4s,
                                     Variable<1, TF> mask,
                                     Variable<1, TF> const& v, TF v_min,
                                     TF v_max, U n) {
    TF step_inverse = static_cast<TF>(
        256.0 / (static_cast<double>(v_max) - static_cast<double>(v_min)));
    launch(
        n,
        [&](U grid_size, U block_size) {
          quantize<<<grid_size, block_size>>>(v, quantized4s, v_min,
                                              step_inverse, n);
        },
        "quantize", quantize<TF>);
    histogram256_with_mask<<<kPartialHistogram256Count,
                             kHistogram256ThreadblockSize>>>(
        partial_histogram, quantized4s, mask, n);
    Allocator::abort_if_error(cudaGetLastError());
    merge_histogram256<<<kHistogram256BinCount, kMergeThreadblockSize>>>(
        histogram, partial_histogram, kPartialHistogram256Count, n);
    Allocator::abort_if_error(cudaGetLastError());
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
      TF3 const& cell_size, TF sign,

      U num_vertices_i) {
    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    // NOTE: InfiniteCylinderDistance and InfiniteTubeDistance omitted
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      launch(
          num_vertices_i,
          [&](U grid_size, U block_size) {
            collision_test<<<grid_size, block_size>>>(
                i, j, vertices_i, num_contacts, contacts, mass_i,
                inertia_tensor_i, x_i, v_i, q_i, omega_i, mass_j,
                inertia_tensor_j, x_j, v_j, q_j, omega_j, restitution, friction,
                distance_nodes, domain_min, domain_max, resolution, cell_size,
                sign, num_vertices_i);
          },
          "collision_test", collision_test<TQ, TF3, TF>);
    } else if (TBoxDistance const* distance =
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
    } else if (TCylinderDistance const* distance =
                   dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
      launch(
          num_vertices_i,
          [&](U grid_size, U block_size) {
            collision_test<<<grid_size, block_size>>>(
                i, j, vertices_i, num_contacts, contacts, mass_i,
                inertia_tensor_i, x_i, v_i, q_i, omega_i, mass_j,
                inertia_tensor_j, x_j, v_j, q_j, omega_j, restitution, friction,
                *distance, sign, num_vertices_i);
          },
          "collision_test(CylinderDistance)",
          collision_test<TQ, TF3, TF, TCylinderDistance>);
    } else if (TCapsuleDistance const* distance =
                   dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
      launch(
          num_vertices_i,
          [&](U grid_size, U block_size) {
            collision_test<<<grid_size, block_size>>>(
                i, j, vertices_i, num_contacts, contacts, mass_i,
                inertia_tensor_i, x_i, v_i, q_i, omega_i, mass_j,
                inertia_tensor_j, x_j, v_j, q_j, omega_j, restitution, friction,
                *distance, sign, num_vertices_i);
          },
          "collision_test(CapsuleDistance)",
          collision_test<TQ, TF3, TF, TCapsuleDistance>);
    } else {
      std::cerr << "[collision_test] Distance type not supported." << std::endl;
    }
  }
  void launch_collision_test_with_pellets(
      dg::Distance<TF3, TF> const& virtual_dist,
      Variable<1, TF> const& distance_nodes, U j_begin, U j_end,
      Variable<1, TF3> const& particle_x,
      Variable<1, U> const& pellet_id_to_rigid_id,

      Variable<1, U>& num_contacts, Variable<1, Contact<TF3, TF>>& contacts,

      Variable<1, TF3> const& x, Variable<1, TF3> const& v,
      Variable<1, TQ> const& q, Variable<1, TF3> const& omega,
      Variable<1, TF3> const& mass_contact, Variable<1, TF3> const& inertia,

      TF3 const& domain_min, TF3 const& domain_max, U3 const& resolution,
      TF3 const& cell_size, TF sign,

      U max_num_beads, U num_pellets) {
    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    // NOTE: InfiniteCylinderDistance and InfiniteTubeDistance omitted
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      launch(
          num_pellets,
          [&](U grid_size, U block_size) {
            collision_test_with_pellets<<<grid_size, block_size>>>(
                j_begin, j_end, particle_x, pellet_id_to_rigid_id, num_contacts,
                contacts, x, v, q, omega, mass_contact, inertia, distance_nodes,
                domain_min, domain_max, resolution, cell_size, sign,
                max_num_beads, num_pellets);
          },
          "collision_test_with_pellets",
          collision_test_with_pellets<TQ, TF3, TF>);
    } else if (TBoxDistance const* distance =
                   dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_pellets,
          [&](U grid_size, U block_size) {
            collision_test_with_pellets<<<grid_size, block_size>>>(
                j_begin, j_end, particle_x, pellet_id_to_rigid_id, num_contacts,
                contacts, x, v, q, omega, mass_contact, inertia, *distance,
                sign, max_num_beads, num_pellets);
          },
          "collision_test_with_pellets(BoxDistance)",
          collision_test_with_pellets<TQ, TF3, TF, TBoxDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_pellets,
          [&](U grid_size, U block_size) {
            collision_test_with_pellets<<<grid_size, block_size>>>(
                j_begin, j_end, particle_x, pellet_id_to_rigid_id, num_contacts,
                contacts, x, v, q, omega, mass_contact, inertia, *distance,
                sign, max_num_beads, num_pellets);
          },
          "collision_test_with_pellets(SphereDistance)",
          collision_test_with_pellets<TQ, TF3, TF, TSphereDistance>);
    } else if (TCylinderDistance const* distance =
                   dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
      launch(
          num_pellets,
          [&](U grid_size, U block_size) {
            collision_test_with_pellets<<<grid_size, block_size>>>(
                j_begin, j_end, particle_x, pellet_id_to_rigid_id, num_contacts,
                contacts, x, v, q, omega, mass_contact, inertia, *distance,
                sign, max_num_beads, num_pellets);
          },
          "collision_test_with_pellets(CylinderDistance)",
          collision_test_with_pellets<TQ, TF3, TF, TCylinderDistance>);
    } else if (TCapsuleDistance const* distance =
                   dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
      launch(
          num_pellets,
          [&](U grid_size, U block_size) {
            collision_test_with_pellets<<<grid_size, block_size>>>(
                j_begin, j_end, particle_x, pellet_id_to_rigid_id, num_contacts,
                contacts, x, v, q, omega, mass_contact, inertia, *distance,
                sign, max_num_beads, num_pellets);
          },
          "collision_test_with_pellets(CapsuleDistance)",
          collision_test_with_pellets<TQ, TF3, TF, TCapsuleDistance>);
    } else if (TInfiniteCylinderDistance const* distance =
                   dynamic_cast<TInfiniteCylinderDistance const*>(
                       &virtual_dist)) {
      // NOTE: remain silent
    } else {
      std::cerr << "[collision_test_with_pellets] Distance type not supported."
                << std::endl;
    }
  }
  template <U wrap>
  void launch_move_and_avoid_boundary(
      Variable<1, TF3>& particle_x, Variable<1, TF3>& particle_dx,
      dg::Distance<TF3, TF> const& virtual_dist,
      Variable<1, TF> const& distance_nodes, TF3 const& domain_min,
      TF3 const& domain_max, U3 const& resolution, TF3 const& cell_size,
      TF sign, TF cfl, U num_particles) {
    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TBoxShellDistance = dg::BoxShellDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    using TInfiniteTubeDistance = dg::InfiniteTubeDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            move_and_avoid_boundary<wrap><<<grid_size, block_size>>>(
                particle_x, particle_dx, distance_nodes, domain_min, domain_max,
                resolution, cell_size, sign, cfl, num_particles);
          },
          "move_and_avoid_boundary", move_and_avoid_boundary<wrap, TF3, TF>);
    } else if (TBoxDistance const* distance =
                   dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            move_and_avoid_boundary<wrap><<<grid_size, block_size>>>(
                particle_x, particle_dx, *distance, sign, cfl, num_particles);
          },
          "move_and_avoid_boundary(BoxDistance)",
          move_and_avoid_boundary<wrap, TF3, TF, TBoxDistance>);
    } else if (TBoxShellDistance const* distance =
                   dynamic_cast<TBoxShellDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            move_and_avoid_boundary<wrap><<<grid_size, block_size>>>(
                particle_x, particle_dx, *distance, sign, cfl, num_particles);
          },
          "move_and_avoid_boundary(BoxShellDistance)",
          move_and_avoid_boundary<wrap, TF3, TF, TBoxShellDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            move_and_avoid_boundary<wrap><<<grid_size, block_size>>>(
                particle_x, particle_dx, *distance, sign, cfl, num_particles);
          },
          "move_and_avoid_boundary(SphereDistance)",
          move_and_avoid_boundary<wrap, TF3, TF, TSphereDistance>);
    } else if (TCylinderDistance const* distance =
                   dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            move_and_avoid_boundary<wrap><<<grid_size, block_size>>>(
                particle_x, particle_dx, *distance, sign, cfl, num_particles);
          },
          "move_and_avoid_boundary(CylinderDistance)",
          move_and_avoid_boundary<wrap, TF3, TF, TCylinderDistance>);
    } else if (TInfiniteCylinderDistance const* distance =
                   dynamic_cast<TInfiniteCylinderDistance const*>(
                       &virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            move_and_avoid_boundary<wrap><<<grid_size, block_size>>>(
                particle_x, particle_dx, *distance, sign, cfl, num_particles);
          },
          "move_and_avoid_boundary(InfiniteCylinderDistance)",
          move_and_avoid_boundary<wrap, TF3, TF, TInfiniteCylinderDistance>);
    } else if (TInfiniteTubeDistance const* distance =
                   dynamic_cast<TInfiniteTubeDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            move_and_avoid_boundary<wrap><<<grid_size, block_size>>>(
                particle_x, particle_dx, *distance, sign, cfl, num_particles);
          },
          "move_and_avoid_boundary(InfiniteTubeDistance)",
          move_and_avoid_boundary<wrap, TF3, TF, TInfiniteTubeDistance>);
    } else if (TCapsuleDistance const* distance =
                   dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            move_and_avoid_boundary<wrap><<<grid_size, block_size>>>(
                particle_x, particle_dx, *distance, sign, cfl, num_particles);
          },
          "move_and_avoid_boundary(CapsuleDistance)",
          move_and_avoid_boundary<wrap, TF3, TF, TCapsuleDistance>);
    } else {
      std::cerr << "[move_and_avoid_boundary] Distance type not supported."
                << std::endl;
    }
  }

  void launch_compute_cohesion_adhesion_displacement(
      Variable<1, TF3>& particle_x, Variable<1, TF>& particle_density,
      Variable<2, TQ>& particle_neighbors,
      Variable<1, U>& particle_num_neighbors, Variable<1, TF3>& particle_dx,
      dg::Distance<TF3, TF> const& virtual_dist,
      Variable<1, TF> const& distance_nodes, TF3 const& domain_min,
      TF3 const& domain_max, U3 const& resolution, TF3 const& cell_size,
      TF sign, TF cohesion, TF adhesion, U num_particles) {
    using TMeshDistance = dg::MeshDistance<TF3, TF>;
    using TBoxDistance = dg::BoxDistance<TF3, TF>;
    using TBoxShellDistance = dg::BoxShellDistance<TF3, TF>;
    using TSphereDistance = dg::SphereDistance<TF3, TF>;
    using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
    using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
    using TInfiniteTubeDistance = dg::InfiniteTubeDistance<TF3, TF>;
    using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
    if (TMeshDistance const* distance =
            dynamic_cast<TMeshDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_cohesion_adhesion_displacement<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_neighbors,
                particle_num_neighbors, particle_dx, distance_nodes, domain_min,
                domain_max, resolution, cell_size, sign, cohesion, adhesion,
                num_particles);
          },
          "compute_cohesion_adhesion_displacement",
          compute_cohesion_adhesion_displacement<TQ, TF3, TF>);
    } else if (TBoxDistance const* distance =
                   dynamic_cast<TBoxDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_cohesion_adhesion_displacement<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_neighbors,
                particle_num_neighbors, particle_dx, *distance, sign, cohesion,
                adhesion, num_particles);
          },
          "compute_cohesion_adhesion_displacement(BoxDistance)",
          compute_cohesion_adhesion_displacement<TQ, TF3, TF, TBoxDistance>);
    } else if (TBoxShellDistance const* distance =
                   dynamic_cast<TBoxShellDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_cohesion_adhesion_displacement<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_neighbors,
                particle_num_neighbors, particle_dx, *distance, sign, cohesion,
                adhesion, num_particles);
          },
          "compute_cohesion_adhesion_displacement(BoxShellDistance)",
          compute_cohesion_adhesion_displacement<TQ, TF3, TF,
                                                 TBoxShellDistance>);
    } else if (TSphereDistance const* distance =
                   dynamic_cast<TSphereDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_cohesion_adhesion_displacement<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_neighbors,
                particle_num_neighbors, particle_dx, *distance, sign, cohesion,
                adhesion, num_particles);
          },
          "compute_cohesion_adhesion_displacement(SphereDistance)",
          compute_cohesion_adhesion_displacement<TQ, TF3, TF, TSphereDistance>);
    } else if (TCylinderDistance const* distance =
                   dynamic_cast<TCylinderDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_cohesion_adhesion_displacement<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_neighbors,
                particle_num_neighbors, particle_dx, *distance, sign, cohesion,
                adhesion, num_particles);
          },
          "compute_cohesion_adhesion_displacement(CylinderDistance)",
          compute_cohesion_adhesion_displacement<TQ, TF3, TF,
                                                 TCylinderDistance>);
    } else if (TInfiniteCylinderDistance const* distance =
                   dynamic_cast<TInfiniteCylinderDistance const*>(
                       &virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_cohesion_adhesion_displacement<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_neighbors,
                particle_num_neighbors, particle_dx, *distance, sign, cohesion,
                adhesion, num_particles);
          },
          "compute_cohesion_adhesion_displacement(InfiniteCylinderDistance)",
          compute_cohesion_adhesion_displacement<TQ, TF3, TF,
                                                 TInfiniteCylinderDistance>);
    } else if (TInfiniteTubeDistance const* distance =
                   dynamic_cast<TInfiniteTubeDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_cohesion_adhesion_displacement<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_neighbors,
                particle_num_neighbors, particle_dx, *distance, sign, cohesion,
                adhesion, num_particles);
          },
          "compute_cohesion_adhesion_displacement(InfiniteTubeDistance)",
          compute_cohesion_adhesion_displacement<TQ, TF3, TF,
                                                 TInfiniteTubeDistance>);
    } else if (TCapsuleDistance const* distance =
                   dynamic_cast<TCapsuleDistance const*>(&virtual_dist)) {
      launch(
          num_particles,
          [&](U grid_size, U block_size) {
            compute_cohesion_adhesion_displacement<<<grid_size, block_size>>>(
                particle_x, particle_density, particle_neighbors,
                particle_num_neighbors, particle_dx, *distance, sign, cohesion,
                adhesion, num_particles);
          },
          "compute_cohesion_adhesion_displacement(CapsuleDistance)",
          compute_cohesion_adhesion_displacement<TQ, TF3, TF,
                                                 TCapsuleDistance>);
    } else {
      std::cerr << "[compute_cohesion_adhesion_displacement] Distance type not "
                   "supported."
                << std::endl;
    }
  }

  using LaunchRecord = std::pair<float, float>;
  constexpr static U kWarpSize = 32;
  constexpr static U kMaxBlockSize = 1024;  // compute capability >= 2
  constexpr static U kNumBlockSizeCandidates = kMaxBlockSize / kWarpSize;
  U default_block_size_;
  std::unordered_map<std::string,
                     std::array<std::pair<U, float>, kNumBlockSizeCandidates>>
      launch_stat_dict_;
  std::unordered_map<std::string, float> custom_elapsed_dict_;
  std::unordered_map<std::string, cudaEvent_t> custom_event_dict_;
  std::unordered_map<std::string, U> optimal_block_size_dict_;
};

}  // namespace alluvion

#endif /* ALLUVION_RUNNER_HPP */
