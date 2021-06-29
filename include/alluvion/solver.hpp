#ifndef ALLUVION_SOLVER_HPP
#define ALLUVION_SOLVER_HPP

namespace alluvion {
template <typename TF>
struct Solver {
  U num_particles;
  TF dt;
  TF max_dt;
  TF min_dt;
  TF cfl;
  TF particle_radius;
};
}  // namespace alluvion

#endif
