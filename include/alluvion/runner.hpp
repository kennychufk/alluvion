#ifndef ALLUVION_RUNNER_HPP
#define ALLUVION_RUNNER_HPP
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

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
  static void sum(void *dst, void *ptr, unsigned int num_elements) {
    *reinterpret_cast<T *>(dst) = thrust::reduce(
        thrust::device_ptr<T>(reinterpret_cast<T *>(ptr)),
        thrust::device_ptr<T>(reinterpret_cast<T *>(ptr)) + num_elements);
  }

  static inline void compute_grid_size(unsigned int n,
                                       unsigned int desired_block_size,
                                       unsigned int &grid_size,
                                       unsigned int &block_size) {
    block_size = std::min(n, desired_block_size);
    grid_size = (n + block_size - 1) / block_size;
  }
};

template <class Lambda>
__device__ void forThreadMappedToElement(unsigned int element_count, Lambda f) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < element_count) {
    f(tid);
  }
}

template <typename TF3, typename TF>
__device__ TF3 make_vector(TF x, TF y, TF z) = delete;

template <>
inline __device__ float3 make_vector<float3>(float x, float y, float z) {
  return make_float3(x, y, z);
}

template <>
inline __device__ double3 make_vector<double3>(double x, double y, double z) {
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

template <typename TF>
__global__ void create_fluid_block(Variable<1, F3> x,
                                   unsigned int num_particles,
                                   unsigned int offset, int mode,
                                   TF particle_radius, TF box_min_x,
                                   TF box_min_y, TF box_min_z, TF box_max_x,
                                   TF box_max_y, TF box_max_z) {
  forThreadMappedToElement(num_particles, [&](unsigned int tid) {
    U p_i = tid + offset;
    TF diameter = particle_radius * 2.0;
    TF eps = 1e-9;
    F3 box_min = make_vector<F3>(box_min_x, box_min_y, box_min_z);
    F3 box_max = make_vector<F3>(box_max_x, box_max_y, box_max_z);
    TF xshift = diameter;
    TF yshift = diameter;
    if (mode == 1) {
      yshift = sqrt(3.0) * particle_radius + eps;
    } else if (mode == 2) {
      xshift = sqrt(6.0) * diameter / 3.0 + eps;
      yshift = sqrt(3.0) * particle_radius + eps;
    }
    F3 diff = box_max - box_min;
    if (mode == 1) {
      diff.x -= diameter;
      diff.z -= diameter;
    } else if (mode == 2) {
      diff.x -= xshift;
      diff.z -= diameter;
    }
    I stepsY = static_cast<I>(diff.y / yshift + 0.5) - 1;
    I stepsZ = static_cast<I>(diff.z / diameter + 0.5) - 1;
    F3 start = box_min + make_vector<F3>(particle_radius * 2);
    I j = p_i / (stepsY * stepsZ);
    I k = (p_i % (stepsY * stepsZ)) / stepsZ;
    I l = p_i % stepsZ;
    F3 currPos = make_vector<F3>(xshift * j, yshift * k, diameter * l) + start;
    F3 shift_vec = make_zeros<F3>();
    if (mode == 1) {
      if (k % 2 == 0) {
        currPos.z += particle_radius;
      } else {
        currPos.x += particle_radius;
      }
    } else if (mode == 2) {
      currPos.z += particle_radius;
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

}  // namespace alluvion

#endif /* ALLUVION_RUNNER_HPP */
