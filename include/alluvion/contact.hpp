#ifndef ALLUVION_CONTACT_HPP
#define ALLUVION_CONTACT_HPP

#include "alluvion/data_type.hpp"

namespace alluvion {
template <typename TF3, typename TF>
struct Contact {
  U i;
  U j;
  TF3 cp_i;
  TF3 cp_j;
  TF3 n;
  TF3 t;
  TF3 iiwi_diag;
  TF3 iiwi_off_diag;
  TF3 iiwj_diag;
  TF3 iiwj_off_diag;
  TF friction;
  TF nkninv;
  TF pmax;
  TF goalu;
  TF impulse_sum;
};
}  // namespace alluvion

#endif /* ALLUVION_CONTACT_HPP */
