#ifndef ALLUVION_CONSTANTS_HPP
#define ALLUVION_CONSTANTS_HPP

#include "alluvion/data_type.hpp"

namespace alluvion {
extern __constant__ F kernel_radius;
void set_kernel_radius(F r);
}  // namespace alluvion

#endif
