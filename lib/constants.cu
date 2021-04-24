#include "alluvion/constants.hpp"

namespace alluvion {
__constant__ F kernel_radius;
void set_kernel_radius(F r) {
  cudaMemcpyToSymbol(&kernel_radius, &r, sizeof(F), 0, cudaMemcpyHostToDevice);
}
}  // namespace alluvion
