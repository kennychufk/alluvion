#include "alluvion/runner.hpp"

namespace alluvion {
Runner::Runner() { cudaEventCreate(&abs_start_); };
Runner::~Runner(){};
}  // namespace alluvion
