#ifndef ALLUVION_CONTACT_HPP
#define ALLUVION_CONTACT_HPP

#include "alluvion/data_type.hpp"

namespace alluvion {
struct Contact {
  U i;
  U j;
  F3 cp_i;
  F3 cp_j;
  F3 n;
  F3 t;
  F3 iiwi_diag;
  F3 iiwi_off_diag;
  F3 iiwj_diag;
  F3 iiwj_off_diag;
  F friction;
  F nkninv;
  F pmax;
  F goalu;
  F impulse_sum;
};
}  // namespace alluvion

#endif /* ALLUVION_CONTACT_HPP */
