#ifndef ALLUVION_USHER_HPP
#define ALLUVION_USHER_HPP

#include <cstring>

#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

namespace alluvion {
template <typename TF>
struct Usher {
 private:
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TPile = Pile<TF>;

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
  std::unique_ptr<Variable<2, TQ>> boundary;
  std::unique_ptr<Variable<2, TQ>> boundary_kernel;
  Store& store;
  U num_ushers;

  Usher(Store& store_arg, TPile& pile_arg, U n)
      : store(store_arg),
        num_ushers(n),
        drive_x(store_arg.create<1, TF3>({n})),
        drive_v(store_arg.create<1, TF3>({n})),
        drive_kernel_radius(store_arg.create<1, TF>({n})),
        drive_strength(store_arg.create<1, TF>({n})),
        neighbors(store_arg.create<2, TQ>(
            {n, store_arg.get_cni().max_num_neighbors_per_particle})),
        num_neighbors(store_arg.create<1, U>({n})),
        sample_x(store_arg.create<1, TF3>({n})),
        sample_v(store_arg.create<1, TF3>({n})),
        sample_density(store_arg.create<1, TF>({n})),
        boundary(store_arg.create<2, TQ>({pile_arg.get_size(), n})),
        boundary_kernel(store_arg.create<2, TQ>({pile_arg.get_size(), n})) {}
  virtual ~Usher() {
    store.remove(*drive_x);
    store.remove(*drive_v);
    store.remove(*drive_kernel_radius);
    store.remove(*drive_strength);

    store.remove(*sample_x);
    store.remove(*neighbors);
    store.remove(*num_neighbors);
    store.remove(*sample_v);
    store.remove(*sample_density);
    store.remove(*boundary);
    store.remove(*boundary_kernel);
  }

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
