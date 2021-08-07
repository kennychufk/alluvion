#ifndef ALLUVION_USHER_HPP
#define ALLUVION_USHER_HPP

#include <cstring>

#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

namespace alluvion {
template <typename TF>
struct Usher {
 private:
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;

 public:
  std::unique_ptr<Variable<1, TF3>> drive_x;
  std::unique_ptr<Variable<1, TF3>> drive_v;
  std::unique_ptr<Variable<1, TF>> drive_kernel_radius;
  std::unique_ptr<Variable<1, TF>> drive_strength;

  // TODO: sample multiple points per usher
  std::unique_ptr<Variable<1, TF3>> sample_x;
  std::unique_ptr<Variable<2, TQ>> neighbors;
  std::unique_ptr<Variable<1, U>> num_neighbors;
  std::unique_ptr<Variable<1, TF3>> sample_v;
  std::unique_ptr<Variable<1, TF>> sample_density;
  std::unique_ptr<Variable<1, TF>> container_dist;
  std::unique_ptr<Variable<1, TF3>> container_normal;
  U num_ushers;

  Usher(Store& store, U n)
      : num_ushers(n),
        drive_x(store.create<1, TF3>({n})),
        drive_v(store.create<1, TF3>({n})),
        drive_kernel_radius(store.create<1, TF>({n})),
        drive_strength(store.create<1, TF>({n})),
        neighbors(store.create<2, TQ>(
            {n, store.get_cni().max_num_neighbors_per_particle})),
        num_neighbors(store.create<1, U>({n})),
        sample_x(store.create<1, TF3>({n})),
        sample_v(store.create<1, TF3>({n})),
        sample_density(store.create<1, TF>({n})),
        container_dist(store.create<1, TF>({n})),
        container_normal(store.create<1, TF3>({n})) {}

  void set(TF3 const* x, TF3 const* v, TF const* r, TF const* strength) {
    drive_x->set_bytes(x);
    drive_v->set_bytes(v);
    drive_kernel_radius->set_bytes(r);
    drive_strength->set_bytes(strength);
  }

  void set_sample_x(TF3 const* x) { sample_x->set_bytes(x); }
};
}  // namespace alluvion

#endif
