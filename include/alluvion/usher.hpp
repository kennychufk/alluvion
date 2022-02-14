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
  std::unique_ptr<Variable<2, TF3>> focal_x;
  std::unique_ptr<Variable<2, TF3>> focal_v;
  std::unique_ptr<Variable<1, TF>> focal_dist;
  std::unique_ptr<Variable<1, TF>> usher_kernel_radius;
  std::unique_ptr<Variable<1, TF>> drive_strength;

  Store& store;
  U num_ushers;

  Usher(Store& store_arg, TPile& pile_arg, U num_ushers_arg)
      : store(store_arg),
        num_ushers(num_ushers_arg),
        focal_x(store_arg.create<2, TF3>({num_ushers_arg, 3})),
        focal_v(store_arg.create<2, TF3>({num_ushers_arg, 3})),
        focal_dist(store_arg.create<1, TF>({num_ushers_arg})),
        usher_kernel_radius(store_arg.create<1, TF>({num_ushers_arg})),
        drive_strength(store_arg.create<1, TF>({num_ushers_arg})) {
    reset();
  }
  virtual ~Usher() {
    store.remove(*focal_x);
    store.remove(*focal_v);
    store.remove(*focal_dist);
    store.remove(*usher_kernel_radius);
    store.remove(*drive_strength);
  }
  void reset() {
    focal_x->set_zero();
    focal_v->set_zero();
    focal_dist->set_zero();
    usher_kernel_radius->set_same(
        64);  // a hack to set non-zero finite floating point values
              // (float: 3.0039215087890625;
              // double: 32.50196078431372370687313377857208251953125)
              // https://baseconvert.com/ieee-754-floating-point
    drive_strength->set_zero();
  }
};
}  // namespace alluvion

#endif
