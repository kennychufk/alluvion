#ifndef ALLUVION_RUNNER_HPP
#define ALLUVION_RUNNER_HPP
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "alluvion/constants.hpp"
#include "alluvion/data_type.hpp"
#include "alluvion/helper_math.h"
#include "alluvion/variable.hpp"
namespace alluvion {
class Runner {
 private:
 public:
  Runner();
  virtual ~Runner();
  template <typename T>
  static void sum(void* dst, void* ptr, U num_elements) {
    *static_cast<T*>(dst) = thrust::reduce(
        thrust::device_ptr<T>(static_cast<T*>(ptr)),
        thrust::device_ptr<T>(static_cast<T*>(ptr)) + num_elements);
  }

  template <class Lambda>
  static void launch(U n, U desired_block_size, Lambda f) {
    U block_size = std::min(n, desired_block_size);
    U grid_size = (n + block_size - 1) / block_size;
    f(grid_size, block_size);
  }
};

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
inline __device__ __host__ float3 make_vector<float3>(float x, float y,
                                                      float z) {
  return make_float3(x, y, z);
}

template <>
inline __device__ __host__ float3 make_vector<float3>(uint x, uint y, uint z) {
  return make_float3(x, y, z);
}

template <>
inline __device__ __host__ double3 make_vector<double3>(double x, double y,
                                                        double z) {
  return make_double3(x, y, z);
}

template <>
inline __device__ __host__ double3 make_vector<double3>(uint x, uint y,
                                                        uint z) {
  return make_double3(x, y, z);
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

template <typename TF3>
__device__ TF3 make_zeros() = delete;

template <>
inline __device__ float3 make_zeros<float3>() {
  return make_float3(0);
}

template <>
inline __device__ double3 make_zeros<double3>() {
  return make_double3(0);
}

template <>
inline __device__ int3 make_zeros<int3>() {
  return make_int3(0);
}

template <typename TQ>
inline __device__ TQ quaternion_conjugate(TQ q) {
  q.x *= -1.0;
  q.y *= -1.0;
  q.z *= -1.0;
  return q;
}

template <typename TF3, typename TQ>
inline __device__ TF3 rotate_using_quaternion(TF3 v, TQ q) {
  TF3 rotated;
  rotated.x = (1.0 - 2.0 * (q.y * q.y + q.z * q.z)) * v.x +
              2.0 * (q.x * q.y - q.z * q.w) * v.y +
              2.0 * (q.x * q.z + q.y * q.w) * v.z;
  rotated.y = 2.0 * (q.x * q.y + q.z * q.w) * v.x +
              (1.0 - 2.0 * (q.x * q.x + q.z * q.z)) * v.y +
              2.0 * (q.y * q.z - q.x * q.w) * v.z;
  rotated.z = 2.0 * (q.x * q.z - q.y * q.w) * v.x +
              2.0 * (q.y * q.z + q.x * q.w) * v.y +
              (1.0 - 2.0 * (q.x * q.x + q.y * q.y)) * v.z;
  return rotated;
}

template <typename TF3>
inline __device__ void get_orthogonal_vectors(TF3 vec, TF3* x, TF3* y) {
  TF3 v{1.0, 0.0, 0.0};
  if (fabs(dot(v, vec)) > 0.999) {
    v.x = 0.0;
    v.y = 1.0;
  }
  *x = cross(vec, v);
  *y = cross(vec, *x);
  *x = normalize(*x);
  *y = normalize(*y);
}

template <typename TF>
inline __device__ TF distance_cubic_kernel(TF r2) {
  TF q2 = r2 / cnst::kernel_radius_sqr;
  TF q = sqrt(r2) / cnst::kernel_radius;
  TF conj = 1.0 - q;
  TF result = 0.0;
  if (q <= 0.5)
    result = (6.0 * q - 6) * q2 + 1.0;
  else if (q <= 1.0)
    result = 2.0 * conj * conj * conj;
  return result * cnst::cubic_kernel_k;
}

template <typename TF3>
inline __device__ F displacement_cubic_kernel(TF3 d) {
  return distance_cubic_kernel(length_sqr(d));
}

template <typename TF3>
inline __device__ TF3 displacement_cubic_kernel_grad(TF3 d) {
  F rl2 = length_sqr(d);
  F rl = sqrt(rl2);
  F q = rl / cnst::kernel_radius;
  TF3 gradq = make_zeros<TF3>();
  if (rl > 1.0e-5) {
    gradq = d / (rl * cnst::kernel_radius);
  }
  TF3 result = make_zeros<TF3>();
  if (q <= 0.5) {
    result = cnst::cubic_kernel_l * q * (3.0 * q - 2.0) * gradq;
  } else if (q <= 1.0 && rl > 1.0e-5) {
    result = cnst::cubic_kernel_l * -(1.0 - q) * (1.0 - q) * gradq;
  }
  return result;
}

template <typename TF>
inline __device__ TF distance_adhesion_kernel(TF r2) {
  TF result = 0.0;
  TF r = sqrt(r2);
  if (r2 < cnst::kernel_radius_sqr && r > 0.5 * cnst::kernel_radius) {
    // TODO:test sqrt(sqrt())
    result = cnst::adhesion_kernel_k * pow(-4 * r2 / cnst::kernel_radius +
                                               6 * r - 2 * cnst::kernel_radius,
                                           0.25);
  }
  return result;
}

template <typename TF3>
inline __device__ F displacement_adhesion_kernel(TF3 d) {
  return distance_adhesion_kernel(length_sqr(d));
}

template <typename TF>
inline __device__ TF distance_cohesion_kernel(TF r2) {
  TF result = 0.0;
  TF r = sqrt(r2);
  TF r3 = r2 * r;
  TF margin = cnst::kernel_radius - r;
  TF margin3 = margin * margin * margin;
  if (r2 <= cnst::kernel_radius_sqr) {
    if (r > cnst::kernel_radius * 0.5) {
      result = cnst::cohesion_kernel_k * margin3 * r3;
    } else {
      result =
          cnst::cohesion_kernel_k * 2 * margin3 * r3 - cnst::cohesion_kernel_c;
    }
  }
  return result;
}

template <typename TF3>
inline __device__ F displacement_cohesion_kernel(TF3 d) {
  return distance_cohesion_kernel(length_sqr(d));
}

template <typename TF3, typename TF>
__global__ void create_fluid_block(Variable<1, TF3> x, U num_particles,
                                   U offset, int mode, TF3 box_min,
                                   TF3 box_max) {
  forThreadMappedToElement(num_particles, [&](U tid) {
    U p_i = tid + offset;
    TF diameter = cnst::particle_radius * 2.0;
    TF eps = 1e-9;
    TF xshift = diameter;
    TF yshift = diameter;
    if (mode == 1) {
      yshift = sqrt(3.0) * cnst::particle_radius + eps;
    } else if (mode == 2) {
      xshift = sqrt(6.0) * diameter / 3.0 + eps;
      yshift = sqrt(3.0) * cnst::particle_radius + eps;
    }
    TF3 diff = box_max - box_min;
    if (mode == 1) {
      diff.x -= diameter;
      diff.z -= diameter;
    } else if (mode == 2) {
      diff.x -= xshift;
      diff.z -= diameter;
    }
    I stepsY = static_cast<I>(diff.y / yshift + 0.5) - 1;
    I stepsZ = static_cast<I>(diff.z / diameter + 0.5) - 1;
    TF3 start = box_min + make_vector<TF3>(cnst::particle_radius * 2);
    I j = p_i / (stepsY * stepsZ);
    I k = (p_i % (stepsY * stepsZ)) / stepsZ;
    I l = p_i % stepsZ;
    TF3 currPos =
        make_vector<TF3>(xshift * j, yshift * k, diameter * l) + start;
    TF3 shift_vec = make_zeros<TF3>();
    if (mode == 1) {
      if (k % 2 == 0) {
        currPos.z += cnst::particle_radius;
      } else {
        currPos.x += cnst::particle_radius;
      }
    } else if (mode == 2) {
      currPos.z += cnst::particle_radius;
      if (j % 2 == 1) {
        if (k % 2 == 0) {
          shift_vec.z = diameter / 2;
        } else {
          shift_vec.z = -diameter / 2;
        }
      }
      if (k % 2 == 0) {
        shift_vec.x = xshift / 2;
      }
    }
    x(offset + p_i) = currPos + shift_vec;
  });
}

template <typename TF3>
__device__ TF3 index_to_node_position(TF3 domain_min, U3 resolution,
                                      TF3 cell_size, U l) {
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
    x.x += (1.0 + (l % 2)) / 3.0 * cell_size.x;
  } else if (l < nv + 2 * (ne_x + ne_y)) {
    l -= (nv + 2 * ne_x);
    e_ind = l / 2;
    temp = e_ind % ((resolution.z + 1) * resolution.y);
    x = domain_min +
        cell_size *
            make_vector<TF3>(e_ind / ((resolution.z + 1) * resolution.y),
                             temp % resolution.y, temp / resolution.y);
    x.y += (1.0 + (l % 2)) / 3.0 * cell_size.y;
  } else {
    l -= (nv + 2 * (ne_x + ne_y));
    e_ind = l / 2;
    temp = e_ind % ((resolution.x + 1) * resolution.z);
    x = domain_min + cell_size * make_vector<TF3>(temp / resolution.z,
                                                  e_ind / ((resolution.x + 1) *
                                                           resolution.z),
                                                  temp % resolution.z);
    x.z += (1.0 + (l % 2)) / 3.0 * cell_size.z;
  }
  return x;
}

template <typename TF3>
__device__ void resolve(TF3 domain_min, TF3 domain_max, U3 resolution,
                        TF3 cell_size, TF3 x, I3* ipos, TF3* inner_x) {
  TF3 sd_min;
  TF3 inv_cell_size = 1.0 / cell_size;
  ipos->x = -1;
  if (x.x > domain_min.x && x.y > domain_min.y && x.z > domain_min.z &&
      domain_max.x > x.x && domain_max.y > x.y && domain_max.z > x.z) {
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

  TF _1mx = 1.0 - xi.x;
  TF _1my = 1.0 - xi.y;
  TF _1mz = 1.0 - xi.z;

  TF _1px = 1.0 + xi.x;
  TF _1py = 1.0 + xi.y;
  TF _1pz = 1.0 + xi.z;

  TF _1m3x = 1.0 - 3.0 * xi.x;
  TF _1m3y = 1.0 - 3.0 * xi.y;
  TF _1m3z = 1.0 - 3.0 * xi.z;

  TF _1p3x = 1.0 + 3.0 * xi.x;
  TF _1p3y = 1.0 + 3.0 * xi.y;
  TF _1p3z = 1.0 + 3.0 * xi.z;

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

  TF _1mx2 = 1.0 - x2;
  TF _1my2 = 1.0 - y2;
  TF _1mz2 = 1.0 - z2;

  // Corner nodes.
  fac = 1.0 / 64.0 * (9.0 * (x2 + y2 + z2) - 19.0);
  shape[0] = fac * _1mxt1my * _1mz;
  shape[1] = fac * _1pxt1my * _1mz;
  shape[2] = fac * _1mxt1py * _1mz;
  shape[3] = fac * _1pxt1py * _1mz;
  shape[4] = fac * _1mxt1my * _1pz;
  shape[5] = fac * _1pxt1my * _1pz;
  shape[6] = fac * _1mxt1py * _1pz;
  shape[7] = fac * _1pxt1py * _1pz;

  // Edge nodes.
  fac = 9.0 / 64.0 * _1mx2;
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

  fac = 9.0 / 64.0 * _1my2;
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

  fac = 9.0 / 64.0 * _1mz2;
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

  TF _1mx = 1.0 - xi.x;
  TF _1my = 1.0 - xi.y;
  TF _1mz = 1.0 - xi.z;

  TF _1px = 1.0 + xi.x;
  TF _1py = 1.0 + xi.y;
  TF _1pz = 1.0 + xi.z;

  TF _1m3x = 1.0 - 3.0 * xi.x;
  TF _1m3y = 1.0 - 3.0 * xi.y;
  TF _1m3z = 1.0 - 3.0 * xi.z;

  TF _1p3x = 1.0 + 3.0 * xi.x;
  TF _1p3y = 1.0 + 3.0 * xi.y;
  TF _1p3z = 1.0 + 3.0 * xi.z;

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

  TF _1mx2 = 1.0 - x2;
  TF _1my2 = 1.0 - y2;
  TF _1mz2 = 1.0 - z2;
  TF _9t3x2py2pz2m19 = 9.0 * (3.0 * x2 + y2 + z2) - 19.0;
  TF _9tx2p3y2pz2m19 = 9.0 * (x2 + 3.0 * y2 + z2) - 19.0;
  TF _9tx2py2p3z2m19 = 9.0 * (x2 + y2 + 3.0 * z2) - 19.0;
  TF _18x = 18.0 * xi.x;
  TF _18y = 18.0 * xi.y;
  TF _18z = 18.0 * xi.z;

  TF _3m9x2 = 3.0 - 9.0 * x2;
  TF _3m9y2 = 3.0 - 9.0 * y2;
  TF _3m9z2 = 3.0 - 9.0 * z2;

  TF _2x = 2.0 * xi.x;
  TF _2y = 2.0 * xi.y;
  TF _2z = 2.0 * xi.z;

  TF _18xm9t3x2py2pz2m19 = _18x - _9t3x2py2pz2m19;
  TF _18xp9t3x2py2pz2m19 = _18x + _9t3x2py2pz2m19;
  TF _18ym9tx2p3y2pz2m19 = _18y - _9tx2p3y2pz2m19;
  TF _18yp9tx2p3y2pz2m19 = _18y + _9tx2p3y2pz2m19;
  TF _18zm9tx2py2p3z2m19 = _18z - _9tx2py2p3z2m19;
  TF _18zp9tx2py2p3z2m19 = _18z + _9tx2py2p3z2m19;

  TF fac0_8 = 1.0 / 64.0;
  TF fac8_32 = 9.0 / 64.0;
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
  fac = 1.0 / 64.0 * (9.0 * (x2 + y2 + z2) - 19.0);
  shape[0] = fac * _1mxt1my * _1mz;
  shape[1] = fac * _1pxt1my * _1mz;
  shape[2] = fac * _1mxt1py * _1mz;
  shape[3] = fac * _1pxt1py * _1mz;
  shape[4] = fac * _1mxt1my * _1pz;
  shape[5] = fac * _1pxt1my * _1pz;
  shape[6] = fac * _1mxt1py * _1pz;
  shape[7] = fac * _1pxt1py * _1pz;

  // Edge nodes.
  fac = 9.0 / 64.0 * _1mx2;
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

  fac = 9.0 / 64.0 * _1my2;
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

  fac = 9.0 / 64.0 * _1mz2;
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
                          F* N) {
  TF phi = 0.0;
  bool max_encountered = false;
  for (U j = 0; j < 32; ++j) {
    TF c = (*nodes)(node_offset + cells[j]);
    max_encountered = (max_encountered || (c == kFMax));
    phi += (!max_encountered) * c * N[j];
  }
  phi = max((max_encountered * 2 - 1) * kFMax, phi);
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
  TF d = kFMax;
  resolve(domain_min, domain_max, resolution, cell_size, x, &ipos, &inner_x);
  if (ipos.x >= 0) {
    get_shape_function(inner_x, N);
    get_cells(resolution, ipos, cells);
    d = interpolate(nodes, node_offset, cells, N);
  }
  return d;
}

template <typename TF>
__global__ void clear_volume_field(Variable<1, TF> volume_nodes, U num_nodes) {
  forThreadMappedToElement(num_nodes, [&](U l) { volume_nodes(l) = 0; });
}

template <typename TF3, typename TF>
__global__ void update_volume_field(Variable<1, TF> volume_nodes,
                                    Variable<1, TF> distance_nodes,
                                    TF3 domain_min, TF3 domain_max,
                                    U3 resolution, TF3 cell_size, U num_nodes,
                                    U node_offset, TF sign, TF thickness) {
  forThreadMappedToElement(num_nodes, [&](U l) {
    for (U i = 0; i < kGridN; ++i) {
      for (U j = 0; j < kGridN; ++j) {
        for (U k = 0; k < kGridN; ++k) {
          TF3 x = index_to_node_position(domain_min, resolution, cell_size, l);
          TF dist = distance_nodes(node_offset + l);
          TF wijk = cnst::kGridWeights[i] * cnst::kGridWeights[j] *
                    cnst::kGridWeights[k];
          TF3 integrand_parameter =
              cnst::kernel_radius * make_vector<TF3>(cnst::kGridAbscissae[i],
                                                     cnst::kGridAbscissae[j],
                                                     cnst::kGridAbscissae[k]);
          TF dist_in_integrand = interpolate_distance_without_intermediates(
              &distance_nodes, domain_min, domain_max, resolution, cell_size,
              node_offset, x + integrand_parameter);
          TF res_in_integrand = 0.0;
          if (dist != kFMax) {
            TF distance_modified =
                (dist - cnst::particle_radius * 0.5 - thickness) * sign;
            if (distance_modified <= cnst::kernel_radius * 2.0) {
              if (length_sqr(integrand_parameter) <= cnst::kernel_radius_sqr) {
                if (dist_in_integrand != kFMax) {
                  TF dist_in_integrand_modified =
                      (dist_in_integrand - cnst::particle_radius * 0.5 -
                       thickness) *
                      sign;
                  if (dist_in_integrand_modified <= 0.0) {
                    res_in_integrand = 1.0 - 0.1 * dist_in_integrand_modified /
                                                 cnst::kernel_radius;
                  } else if (dist_in_integrand_modified < cnst::kernel_radius) {
                    res_in_integrand =
                        distance_cubic_kernel(dist_in_integrand_modified *
                                              dist_in_integrand_modified) /
                        cnst::cubic_kernel_zero;
                  }
                }
              }
            }
          }
          atomicAdd(&volume_nodes(node_offset + l),
                    wijk * 0.8 * res_in_integrand * cnst::kernel_radius_sqr *
                        cnst::kernel_radius);
        }
      }
    }
  });
}

template <typename TQ, typename TF3, typename TF>
__device__ TF compute_volume_and_boundary_x(
    Variable<1, TF>* volume_nodes, Variable<1, TF>* distance_nodes,
    TF3 domain_min, TF3 domain_max, U3 resolution, TF3 cell_size, U num_nodes,
    U node_offset, TF sign, TF thickness, TF3& x, TF3& rigid_x, TQ& rigid_q,
    TF dt, TF3* boundary_xj, TF* d, TF3* normal) {
  TF boundary_volume = 0.0;
  TF3 shifted = x - rigid_x;
  TF3 local_xi =
      rotate_using_quaternion(shifted, quaternion_conjugate(rigid_q));
  // for resolve
  I3 ipos;
  TF3 inner_x;
  // for get_shape_function_and_gradient
  TF N[32];
  TF dN0[32];
  TF dN1[32];
  TF dN2[32];
  U cells[32];
  // for interpolate_and_derive
  TF dist;
  // other
  TF volume, nl;
  bool write_boundary, penetrated;
  *boundary_xj = make_zeros<TF3>();
  *d = 0.0;
  *normal = make_zeros<TF3>();

  resolve(domain_min, domain_max, resolution, cell_size, local_xi, &ipos,
          &inner_x);
  if (ipos.x >= 0) {
    get_shape_function_and_gradient(inner_x, N, dN0, dN1, dN2);
    get_cells(resolution, ipos, cells);
    dist = interpolate_and_derive(distance_nodes, node_offset, &cell_size,
                                  cells, N, dN0, dN1, dN2, normal);
    *normal *= sign;
    dist = (dist - cnst::particle_radius * 0.5 - thickness) * sign;
    *normal = rotate_using_quaternion(*normal, rigid_q);
    nl = length(*normal);
    *normal /= nl;
    volume = interpolate(volume_nodes, node_offset, cells, N);
    write_boundary =
        (dist > 0.1 * cnst::particle_radius && dist < cnst::kernel_radius &&
         volume > 1e-5 && volume != kFMax && nl > 1e-5);
    boundary_volume = write_boundary * volume;
    *boundary_xj = write_boundary * (x - dist * (*normal));
    penetrated = (dist <= 0.1 * cnst::particle_radius && nl > 1e-5);
    *d = penetrated * -dist;
    *d = min(*d, 0.25 / 0.005 * cnst::particle_radius * dt);
  }
  return boundary_volume;
}

// gradient must be initialized to zero
template <typename TF3, typename TF>
__device__ TF interpolate_and_derive(Variable<1, TF>* nodes, U node_offset,
                                     TF3* cell_size, U* cells, F* N, TF* dN0,
                                     TF* dN1, TF* dN2, TF3* gradient) {
  TF phi = 0.0;
  bool max_encountered = false;
  for (U j = 0; j < 32; ++j) {
    TF c = (*nodes)(node_offset + cells[j]);
    max_encountered = (max_encountered || (c == kFMax));
    phi += (!max_encountered) * c * N[j];
    gradient->x += (not max_encountered) * c * dN0[j];
    gradient->y += (not max_encountered) * c * dN1[j];
    gradient->z += (not max_encountered) * c * dN2[j];
  }
  gradient->x *= (not max_encountered) * 2 / cell_size->x;
  gradient->y *= (not max_encountered) * 2 / cell_size->y;
  gradient->z *= (not max_encountered) * 2 / cell_size->z;
  phi = max((max_encountered * 2 - 1) * kFMax, phi);
  return phi;
}

template <typename TF3, typename TF>
__device__ TF collision_find_dist_normal(Variable<1, TF> distance_nodes,
                                         TF3 domain_min, TF3 domain_max,
                                         U3 resolution, TF3 cell_size,
                                         U node_offset, TF sign, TF tolerance,
                                         TF3 x, TF3* normal) {
  TF dist = 0.0;
  I3 ipos;
  TF3 inner_x;
  TF N[32];
  TF dN0[32];
  TF dN1[32];
  TF dN2[32];
  U cells[32];
  TF d = kFMax;
  TF3 normal_tmp = make_zeros<TF3>();
  *normal = make_zeros<TF3>();

  resolve(domain_min, domain_max, resolution, cell_size, x, &ipos, &inner_x);
  if (ipos.x >= 0) {
    get_shape_function_and_gradient(inner_x, N, dN0, dN1, dN2);
    get_cells(resolution, ipos, cells);
    d = interpolate_and_derive(distance_nodes, node_offset, &cell_size, cells,
                               N, dN0, dN1, dN2, normal_tmp);
  }
  if (d != kFMax) {
    dist = sign * d - tolerance;
    normal_tmp *= sign;
  }
  if (dist < 0.0) {
    *normal = length(normal_tmp);
  }
  return dist;
}

// fluid neighbor list
template <typename TF3>
__device__ I3 get_ipos(TF3 x) {
  return make_int3(floor(x / cnst::cell_width)) - cnst::grid_offset;
}

template <typename TI3>
__device__ bool within_grid(TI3 ipos) {
  return 0 <= ipos.x and ipos.x < cnst::grid_res.x and 0 <= ipos.y and
         ipos.y < cnst::grid_res.y and 0 <= ipos.z and
         ipos.z < cnst::grid_res.z;
}

template <typename TU>
__global__ void clear_particle_grid(Variable<3, TU> pid_length,
                                    TU num_grid_cells) {
  forThreadMappedToElement(num_grid_cells, [&](U i) { pid_length(i) = 0; });
}

template <typename TF3>
__global__ void update_particle_grid(Variable<1, TF3> particle_x,
                                     Variable<4, U> pid,
                                     Variable<3, U> pid_length,
                                     U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    I3 ipos = get_ipos(particle_x(p_i));
    U pid_insert_index;
    if (within_grid(ipos)) {
      pid_insert_index = atomicAdd(&pid_length(ipos), 1);
      if (pid_insert_index == cnst::max_num_particles_per_cell) {
        printf("Too many particles at ipos = (%d, %d, %d)\n", ipos.x, ipos.y,
               ipos.z);
      }
      pid(ipos, pid_insert_index) = p_i;
    } else {
      printf("Particle falls out of the grid\n");
    }
  });
}

template <typename TF3>
__global__ void make_neighbor_list(Variable<1, TF3> particle_x,
                                   Variable<4, U> pid,
                                   Variable<3, U> pid_length,
                                   Variable<2, U> particle_neighbors,
                                   Variable<1, U> particle_num_neighbors,
                                   U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x = particle_x(p_i);
    I3 ipos = get_ipos(x);
    U num_neighbors = 0;
    for (U i = 0; i < cnst::num_cells_to_search; ++i) {
      I3 neighbor_ipos = ipos + cnst::neighbor_offsets[i];
      if (within_grid(neighbor_ipos)) {
        U neighbor_occupancy =
            min(pid_length(neighbor_ipos), cnst::max_num_particles_per_cell);
        for (U k = 0; k < neighbor_occupancy; ++k) {
          U p_j = pid(neighbor_ipos, k);
          if (p_j != p_i &&
              length_sqr(x - particle_x(p_j)) < cnst::kernel_radius_sqr) {
            particle_neighbors(p_i, num_neighbors) = p_j;
            num_neighbors += 1;
          }
        }
      }
    }
    particle_num_neighbors(p_i) = num_neighbors;
  });
}

// fluid
template <typename TF3>
__global__ void clear_acceleration(Variable<1, TF3> particle_a,
                                   U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_a(p_i) = make_vector<TF3>(static_cast<F>(0.0), cnst::gravity,
                                       static_cast<F>(0.0));
  });
}

template <typename TQ, typename TF3, typename TF>
__global__ void compute_particle_boundary(
    Variable<1, TF> volume_nodes, Variable<1, TF> distance_nodes, TF3 rigid_x,
    TQ rigid_q, U boundary_id, TF3 domain_min, TF3 domain_max, U3 resolution,
    TF3 cell_size, U num_nodes, U node_offset, TF sign, TF thickness, TF dt,
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<2, TF3> particle_boundary_xj,
    Variable<2, TF> particle_boundary_volume, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 boundary_xj, normal;
    TF d;
    bool penetrated;
    TF boundary_volume = compute_volume_and_boundary_x(
        &volume_nodes, &distance_nodes, domain_min, domain_max, resolution,
        cell_size, num_nodes, node_offset, sign, thickness, particle_x(p_i),
        rigid_x, rigid_q, dt, &boundary_xj, &d, &normal);
    particle_boundary_xj(boundary_id, p_i) = boundary_xj;
    particle_boundary_volume(boundary_id, p_i) = boundary_volume;
    penetrated = (d != 0.0);
    particle_x(p_i) += penetrated * d * normal;
    particle_v(p_i) +=
        penetrated * (0.05 - dot(particle_v(p_i), normal)) * normal;
  });
}

template <typename TF3, typename TF>
__global__ void compute_density_fluid(Variable<1, TF3> particle_x,
                                      Variable<2, U> particle_neighbors,
                                      Variable<1, U> particle_num_neighbors,
                                      Variable<1, TF> particle_density,
                                      U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF density = cnst::particle_vol * cnst::cubic_kernel_zero;
    TF3 x_i = particle_x(p_i);
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 x_j = particle_x(p_j);
      density += cnst::particle_vol * displacement_cubic_kernel(x_i - x_j);
    }
    particle_density(p_i) = density * cnst::density0;
  });
}

template <typename TF3, typename TF>
__global__ void compute_density_boundary(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<2, TF3> particle_boundary_xj,
    Variable<2, TF> particle_boundary_volume, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF density = 0.0;
    TF3 x_i = particle_x(p_i);
    for (U boundary_id = 0; boundary_id < cnst::num_boundaries; ++boundary_id) {
      TF vj = max(0.0, particle_boundary_volume(boundary_id, p_i));
      TF3 bx_j = particle_boundary_xj(boundary_id, p_i);
      density += vj * displacement_cubic_kernel(x_i - bx_j);
    }
    particle_density(p_i) += density * cnst::density0;
  });
}

// Viscosity_Standard
template <typename TF3, typename TF>
__global__ void compute_viscosity_fluid(Variable<1, TF3> particle_x,
                                        Variable<1, TF3> particle_v,
                                        Variable<1, TF> particle_density,
                                        Variable<2, U> particle_neighbors,
                                        Variable<1, U> particle_num_neighbors,
                                        Variable<1, TF3> particle_a,
                                        U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF d = 10.0;
    TF3 v_i = particle_v(p_i);
    TF3 x_i = particle_x(p_i);
    TF3 da = make_zeros<TF3>();

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 x_j = particle_x(p_j);
      TF3 xixj = x_i - x_j;
      da += d * cnst::viscosity * cnst::particle_mass / particle_density(p_j) *
            dot(v_i - particle_v(p_j), xixj) /
            (length_sqr(xixj) + 0.01 * cnst::kernel_radius_sqr) *
            displacement_cubic_kernel_grad(xixj);
    }
    particle_a(p_i) += da;
  });
}

template <typename TF3, typename TF>
__global__ void compute_viscosity_boundary(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF3> particle_a, Variable<2, TF3> particle_force,
    Variable<2, TF3> particle_torque, Variable<2, TF3> particle_boundary_xj,
    Variable<2, TF> particle_boundary_volume, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega,
    Variable<1, TF> boundary_viscosity, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF d = 10.0;
    TF3 v_i = particle_v(p_i);
    TF3 x_i = particle_x(p_i);
    TF3 da = make_zeros<TF3>();
    for (U boundary_id = 0; boundary_id < cnst::num_boundaries; ++boundary_id) {
      TF vj = max(0.0, particle_boundary_volume(boundary_id, p_i));
      TF3 bx_j = particle_boundary_xj(boundary_id, p_i);
      TF3 r_x = rigid_x(boundary_id);
      TF3 r_v = rigid_v(boundary_id);
      TF3 r_omega = rigid_omega(boundary_id);
      TF b_viscosity = boundary_viscosity(boundary_id);

      TF3 normal = bx_j - x_i;
      TF nl = length(normal);
      if (nl > 0.0001) {
        normal /= nl;
        TF3 t1, t2;
        get_orthogonal_vectors(normal, &t1, &t2);

        TF dist = (1.0 - nl / cnst::kernel_radius) * cnst::kernel_radius;
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
        TF vol = 0.25 * vj;

        TF3 v1 = cross(r_omega, x1 - r_x) + r_v;
        TF3 v2 = cross(r_omega, x2 - r_x) + r_v;
        TF3 v3 = cross(r_omega, x3 - r_x) + r_v;
        TF3 v4 = cross(r_omega, x4 - r_x) + r_v;

        // compute forces for both sample point
        TF3 a1 = d * b_viscosity * vol * dot(v_i - v1, xix1) /
                 (length_sqr(xix1) + 0.01 * cnst::kernel_radius_sqr) * gradW1;
        TF3 a2 = d * b_viscosity * vol * dot(v_i - v2, xix2) /
                 (length_sqr(xix2) + 0.01 * cnst::kernel_radius_sqr) * gradW2;
        TF3 a3 = d * b_viscosity * vol * dot(v_i - v3, xix3) /
                 (length_sqr(xix3) + 0.01 * cnst::kernel_radius_sqr) * gradW3;
        TF3 a4 = d * b_viscosity * vol * dot(v_i - v4, xix4) /
                 (length_sqr(xix4) + 0.01 * cnst::kernel_radius_sqr) * gradW4;
        da += a1 + a2 + a3 + a4;

        TF3 f1 = -cnst::particle_mass * a1;
        TF3 f2 = -cnst::particle_mass * a2;
        TF3 f3 = -cnst::particle_mass * a3;
        TF3 f4 = -cnst::particle_mass * a4;
        TF3 torque1 = cross(x1 - r_x, f1);
        TF3 torque2 = cross(x2 - r_x, f2);
        TF3 torque3 = cross(x3 - r_x, f3);
        TF3 torque4 = cross(x4 - r_x, f4);

        particle_force(boundary_id, p_i) = f1 + f2 + f3 + f4;
        particle_torque(boundary_id, p_i) =
            torque1 + torque2 + torque3 + torque4;
      }
    }
    particle_a(p_i) += da;
  });
}

template <typename TF3, typename TF>
__global__ void reset_angular_acceleration(
    Variable<1, TF3> particle_angular_acceleration, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_angular_acceleration(p_i) = make_zeros<TF3>();
  });
}

// Micropolar Model
template <typename TF3, typename TF>
__global__ void compute_vorticity_fluid(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<1, TF3> particle_omega,
    Variable<1, TF3> particle_a, Variable<1, TF3> particle_angular_acceleration,
    Variable<2, U> particle_neighbors, TF dt,
    Variable<1, U> particle_num_neighbors, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v_i = particle_v(p_i);
    TF3 omegai = particle_omega(p_i);
    TF3 density_i = particle_density(p_i);

    TF3 da = make_zeros<TF3>();
    TF3 dangular_acc =
        -2 * cnst::inertia_inverse * cnst::vorticity_coeff * omegai;

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 xixj = x_i - particle_x(p_j);
      TF3 omegaij = omegai - particle_omega(p_j);
      TF3 grad_w = displacement_cubic_kernel_grad(xixj);

      dangular_acc -= cnst::inertia_inverse * cnst::viscosity_omega / dt *
                      cnst::particle_mass / particle_density(p_j) * omegaij *
                      displacement_cubic_kernel(xixj);
      da += cnst::vorticity_coeff / density_i * cnst::particle_mass *
            cross(omegaij, grad_w);
      dangular_acc +=
          cnst::vorticity_coeff / density_i * cnst::inertia_inverse *
          cross(cnst::particle_mass * (v_i - particle_v(p_j)), grad_w);
    }

    particle_a(p_i) += da;
    particle_angular_acceleration(p_i) += dangular_acc;
  });
}
template <typename TF3, typename TF>
__global__ void compute_vorticity_boundary(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<1, TF3> particle_omega,
    Variable<1, TF3> particle_a, Variable<1, TF3> particle_angular_acceleration,
    Variable<2, TF3> particle_boundary_xj,
    Variable<2, TF> particle_boundary_volume, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v_i = particle_v(p_i);
    TF3 omegai = particle_omega(p_i);
    TF density_i = particle_density(p_i);

    TF3 da = make_zeros<TF3>();
    TF3 dangular_acc = make_zeros<TF3>();
    for (U boundary_id = 0; boundary_id < cnst::num_boundaries; ++boundary_id) {
      TF vj = max(0.0, particle_boundary_volume(boundary_id, p_i));
      TF3 bx_j = particle_boundary_xj(boundary_id, p_i);
      TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
                 rigid_v(boundary_id);
      TF3 omegaij = omegai;  // omegaj not implemented in SPlisHSPlasH
      TF3 xixj = x_i - bx_j;
      TF3 grad_w = displacement_cubic_kernel_grad(xixj);
      da += cnst::vorticity_coeff / density_i * cnst::density0 * vj *
            cross(omegaij, grad_w);
      dangular_acc += cnst::vorticity_coeff / density_i *
                      cnst::inertia_inverse *
                      cross(cnst::density0 * vj * (v_i - velj), grad_w);
    }
    particle_a(p_i) += da;
    particle_angular_acceleration(p_i) += dangular_acc;
  });
}
template <typename TF3, typename TF>
__global__ void integrate_angular_acceleration(
    Variable<1, TF3> particle_omega,
    Variable<1, TF3> particle_angular_acceleration, TF dt, U num_neighbors,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_omega(p_i) += dt * particle_angular_acceleration(p_i);
  });
}
template <typename TF3, typename TF>
__global__ void compute_normal(Variable<1, TF3> particle_x,
                               Variable<1, TF> particle_density,
                               Variable<1, TF3> particle_normal,
                               Variable<2, U> particle_neighbors,
                               Variable<1, U> particle_num_neighbors,
                               U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 ni = make_zeros<TF3>();
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 x_j = particle_x(p_j);
      ni += cnst::particle_mass / particle_density(p_j) *
            displacement_cubic_kernel_grad(x_i - x_j);
    }
    particle_normal(p_i) = ni;
  });
}
template <typename TF3, typename TF>
__global__ void compute_surface_tension_fluid(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF3> particle_normal, Variable<1, TF3> particle_a,
    Variable<2, U> particle_neighbors, Variable<1, U> particle_num_neighbors,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 ni = particle_normal(p_i);
    TF density_i = particle_density(p_i);

    TF3 da = make_zeros<TF3>();
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 x_j = particle_x(p_j);
      TF k_ij = cnst::density0 * 2 / (density_i + particle_density(p_j));
      TF3 xixj = x_i - x_j;
      TF length2 = length_sqr(xixj);
      TF3 accel = make_zeros<TF3>();
      if (length2 > 1e-9) {
        accel = -cnst::surface_tension_coeff * cnst::particle_mass *
                displacement_cohesion_kernel(xixj) / sqrt(length2) * xixj;
      }
      accel -= cnst::surface_tension_coeff * (ni - particle_normal(p_j));
      da += k_ij * accel;
    }
    particle_a(p_i) += da;
  });
}

template <typename TF3, typename TF>
__global__ void compute_surface_tension_boundary(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF3> particle_a, Variable<1, TF3> particle_normal,
    Variable<2, TF3> particle_boundary_xj,
    Variable<2, TF> particle_boundary_volume, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 ni = particle_normal(p_i);
    TF density_i = particle_density(p_i);

    TF3 da = make_zeros<TF3>();
    for (U boundary_id = 0; boundary_id < cnst::num_boundaries; ++boundary_id) {
      TF vj = max(0.0, particle_boundary_volume(boundary_id, p_i));
      TF3 bx_j = particle_boundary_xj(boundary_id, p_i);
      TF3 xixj = x_i - bx_j;
      TF length2 = length_sqr(xixj);
      if (length2 > 1e-9) {
        da -= cnst::surface_tension_boundary_coeff * vj * cnst::density0 *
              displacement_adhesion_kernel(xixj) / sqrt(length2) * xixj;
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
    particle_cfl_v2(p_i) += length_sqr(particle_v(p_i) + particle_a(p_i) * dt) /
                            num_particles / num_particles;
  });
}
template <typename TF3, typename TF>
__global__ void predict_advection0_fluid_advect(Variable<1, TF3> particle_v,
                                                Variable<1, TF3> particle_a,
                                                TF dt, U num_particles) {
  forThreadMappedToElement(
      num_particles, [&](U p_i) { particle_v(p_i) += dt * particle_a(p_i); });
}
template <typename TF3, typename TF>
__global__ void predict_advection0_fluid(Variable<1, TF3> particle_x,
                                         Variable<1, TF> particle_density,
                                         Variable<1, TF3> particle_dii,
                                         Variable<2, U> particle_neighbors,
                                         Variable<1, U> particle_num_neighbors,
                                         U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF density = particle_density(p_i) / cnst::density0;
    TF density2 = density * density;

    TF3 dii = make_zeros<TF3>();
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 x_j = particle_x(p_j);
      dii -= cnst::particle_vol / density2 *
             displacement_cubic_kernel_grad(x_i - x_j);
    }
    particle_dii(p_i) = dii;
  });
}
template <typename TF3, typename TF>
__global__ void predict_advection0_boundary(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF3> particle_dii, Variable<2, TF3> particle_boundary_xj,
    Variable<2, TF> particle_boundary_volume, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF density = particle_density(p_i) / cnst::density0;
    TF density2 = density * density;
    TF3 dii = make_zeros<TF3>();
    for (U boundary_id = 0; boundary_id < cnst::num_boundaries; ++boundary_id) {
      TF vj = max(0.0, particle_boundary_volume(boundary_id, p_i));
      TF3 bx_j = particle_boundary_xj(boundary_id, p_i);
      dii -= vj / density2 * displacement_cubic_kernel_grad(x_i - bx_j);
    }
    particle_dii(p_i) += dii;
  });
}
template <typename TF>
__global__ void reset_last_pressure(Variable<1, TF> particle_pressure,
                                    Variable<1, TF> particle_last_pressure,
                                    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    particle_last_pressure(p_i) = particle_pressure(p_i) * 0.5;
  });
}
template <typename TF3, typename TF>
__global__ void predict_advection1_fluid(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF3> particle_dii, Variable<1, TF> particle_adv_density,
    Variable<1, TF> particle_aii, Variable<1, TF> particle_density,
    Variable<2, U> particle_neighbors, Variable<1, U> particle_num_neighbors,
    TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v = particle_v(p_i);
    TF3 dii = particle_dii(p_i);
    TF density = particle_density(p_i) / cnst::density0;
    TF density2 = density * density;
    TF dpi = cnst::particle_vol / density2;

    // target
    TF density_adv = density;
    TF aii = 0.0;

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 x_j = particle_x(p_j);

      TF3 grad_w = displacement_cubic_kernel_grad(x_i - x_j);
      TF3 dji = dpi * grad_w;

      density_adv += dt * cnst::particle_vol * dot(v - particle_v(p_j), grad_w);
      aii += cnst::particle_vol * dot(dii - dji, grad_w);
    }
    particle_adv_density(p_i) = density_adv;
    particle_aii(p_i) = aii;
  });
}
template <typename TF3, typename TF>
__global__ void predict_advection1_boundary(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_v,
    Variable<1, TF> particle_density, Variable<1, TF3> particle_dii,
    Variable<1, TF> particle_adv_density, Variable<1, TF> particle_aii,
    Variable<2, TF3> particle_boundary_xj,
    Variable<2, TF> particle_boundary_volume, Variable<1, TF3> rigid_x,
    Variable<1, TF3> rigid_v, Variable<1, TF3> rigid_omega, TF dt,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v = particle_v(p_i);
    TF3 dii = particle_dii(p_i);
    TF density = particle_density(p_i) / cnst::density0;
    TF density2 = density * density;
    TF dpi = cnst::particle_vol / density2;

    // target
    TF density_adv = 0.0;
    TF aii = 0.0;

    for (U boundary_id = 0; boundary_id < cnst::num_boundaries; ++boundary_id) {
      TF vj = max(0.0, particle_boundary_volume(boundary_id, p_i));
      TF3 bx_j = particle_boundary_xj(boundary_id, p_i);

      TF3 velj = cross(rigid_omega(boundary_id), bx_j - rigid_x(boundary_id)) +
                 rigid_v(boundary_id);
      TF3 grad_w = displacement_cubic_kernel_grad(x_i - bx_j);
      TF3 dji = dpi * grad_w;
      density_adv += dt * vj * dot(v - velj, grad_w);
      aii += vj * dot(dii - dji, grad_w);
    }
    particle_adv_density(p_i) += density_adv;
    particle_aii(p_i) += aii;
  });
}
template <typename TF3, typename TF>
__global__ void pressure_solve_iteration0(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_last_pressure, Variable<1, TF3> particle_dij_pj,
    Variable<2, U> particle_neighbors, Variable<1, U> particle_num_neighbors,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);

    // target
    TF3 dij_pj = make_zeros<TF3>();
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 x_j = particle_x(p_j);

      TF densityj = particle_density(p_j) / cnst::density0;
      TF densityj2 = densityj * densityj;
      TF last_pressure = particle_last_pressure(p_j);

      dij_pj -= cnst::particle_vol / densityj2 * last_pressure *
                displacement_cubic_kernel_grad(x_i - x_j);
    }
    particle_dij_pj(p_i) = dij_pj;
  });
}
template <typename TF3, typename TF>
__global__ void pressure_solve_iteration1_fluid(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_last_pressure, Variable<1, TF3> particle_dii,
    Variable<1, TF3> particle_dij_pj, Variable<1, TF> particle_sum_tmp,
    Variable<2, U> particle_neighbors, Variable<1, U> particle_num_neighbors,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF last_pressure = particle_last_pressure(p_i);
    TF3 dij_pj = particle_dij_pj(p_i);
    TF density = particle_density(p_i) / cnst::density0;
    TF density2 = density * density;
    TF dpi = cnst::particle_vol / density2;

    TF sum_tmp = 0.0;
    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 x_j = particle_x(p_j);

      TF3 djk_pk = particle_dij_pj(p_j);
      TF3 grad_w = displacement_cubic_kernel_grad(x_i - x_j);
      TF3 dji = dpi * grad_w;
      TF3 dji_pi = dji * last_pressure;

      sum_tmp += cnst::particle_vol *
                 dot(dij_pj - particle_dii(p_j) * particle_last_pressure(p_j) -
                         (djk_pk - dji_pi),
                     grad_w);
    }
    particle_sum_tmp(p_i) = sum_tmp;
  });
}
template <typename TF3, typename TF>
__global__ void pressure_solve_iteration1_boundary(
    Variable<1, TF3> particle_x, Variable<1, TF3> particle_dij_pj,
    Variable<1, TF> particle_sum_tmp, Variable<2, TF3> particle_boundary_xj,
    Variable<2, TF> particle_boundary_volume, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 dij_pj = particle_dij_pj(p_i);

    TF sum_tmp = 0.0;
    for (U boundary_id = 0; boundary_id < cnst::num_boundaries; ++boundary_id) {
      TF vj = max(0.0, particle_boundary_volume(boundary_id, p_i));
      TF3 bx_j = particle_boundary_xj(boundary_id, p_i);

      sum_tmp += vj * dot(dij_pj, displacement_cubic_kernel_grad(x_i - bx_j));
    }
    particle_sum_tmp(p_i) += sum_tmp;
  });
}
template <typename TF>
__global__ void pressure_solve_iteration1_summarize(
    Variable<1, TF> particle_aii, Variable<1, TF> particle_adv_density,
    Variable<1, TF> particle_sum_tmp, Variable<1, TF> particle_last_pressure,
    Variable<1, TF> particle_pressure, TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF last_pressure = particle_last_pressure(p_i);
    TF b = 1.0 - particle_adv_density(p_i);
    TF dt2 = dt * dt;
    TF aii = particle_aii(p_i);
    TF denom = aii * dt2;
    TF omega = 0.5;
    TF sum_tmp = particle_sum_tmp(p_i);

    TF pressure = 0.0;
    if (fabs(denom) > 1.0e-9) {
      pressure = max(
          (1.0 - omega) * last_pressure + omega / denom * (b - dt2 * sum_tmp),
          0.0);
    }
    particle_pressure(p_i) = pressure;
    particle_last_pressure(p_i) = pressure;
  });
}

template <typename TF3, typename TF>
__global__ void compute_pressure_accels_fluid(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_pressure, Variable<1, TF3> particle_pressure_accel,
    Variable<2, U> particle_neighbors, Variable<1, U> particle_num_neighbors,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF density = particle_density(p_i) / cnst::density0;
    TF density2 = density * density;
    TF dpi = particle_pressure(p_i) / density2;

    // target
    TF3 ai = make_zeros<TF3>();

    for (U neighbor_id = 0; neighbor_id < particle_num_neighbors(p_i);
         ++neighbor_id) {
      U p_j = particle_neighbors(p_i, neighbor_id);
      TF3 x_j = particle_x(p_j);

      TF densityj = particle_density(p_j) / cnst::density0;
      TF densityj2 = densityj * densityj;
      TF dpj = particle_pressure(p_j) / densityj2;
      ai -= cnst::particle_vol * (dpi + dpj) *
            displacement_cubic_kernel_grad(x_i - x_j);
    }
    particle_pressure_accel(p_i) = ai;
  });
}
template <typename TF3, typename TF>
__global__ void compute_pressure_accels_boundary(
    Variable<1, TF3> particle_x, Variable<1, TF> particle_density,
    Variable<1, TF> particle_pressure, Variable<1, TF3> particle_pressure_accel,
    Variable<2, TF3> particle_force, Variable<2, TF3> particle_torque,
    Variable<2, TF3> particle_boundary_xj,
    Variable<2, TF> particle_boundary_volume, Variable<1, TF3> rigid_x,
    U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF density = particle_density(p_i) / cnst::density0;
    TF density2 = density * density;
    TF dpi = particle_pressure(p_i) / density2;

    // target
    TF3 ai = make_zeros<TF3>();

    for (U boundary_id = 0; boundary_id < cnst::num_boundaries; ++boundary_id) {
      TF vj = max(0.0, particle_boundary_volume(boundary_id, p_i));
      TF3 bx_j = particle_boundary_xj(boundary_id, p_i);
      TF3 a = vj * dpi * displacement_cubic_kernel_grad(x_i - bx_j);
      TF3 force = cnst::particle_mass * a;
      ai -= a;
      particle_force(boundary_id, p_i) = force;
      particle_torque(boundary_id, p_i) =
          cross(bx_j - rigid_x(boundary_id), force);
    }
    particle_pressure_accel(p_i) += ai;
  });
}
template <typename TF3, typename TF>
__global__ void kinematic_integration(Variable<1, TF3> particle_x,
                                      Variable<1, TF3> particle_v,
                                      Variable<1, TF3> particle_pressure_accel,
                                      TF dt, U num_particles) {
  forThreadMappedToElement(num_particles, [&](U p_i) {
    TF3 x_i = particle_x(p_i);
    TF3 v = particle_v(p_i);
    v += particle_pressure_accel(p_i) * dt;
    x_i += v * dt;
    // TODO: refactor into particle_x(p_i)+= v * dt;
    particle_x(p_i) = x_i;
    particle_v(p_i) = v;
  });
}

}  // namespace alluvion

#endif /* ALLUVION_RUNNER_HPP */
