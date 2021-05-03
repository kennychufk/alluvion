#ifndef ALLUVION_DG_VOLUME_INTEGRATION_HPP
#define ALLUVION_DG_VOLUME_INTEGRATION_HPP

#include <iostream>
#include <string>
#include <vector>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {
class VolumeIntegration {
 private:
  int A;
  int B;
  int C;

  // projection integrals
  F P1, Pa, Pb, Paa, Pab, Pbb, Paaa, Paab, Pabb, Pbbb;
  // face integrals
  F Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca;
  // volume integrals
  F T0;
  F T1[3];
  F T2[3];
  F TP[3];

 public:
  VolumeIntegration(std::vector<Vector3r> const& vertices,
                    std::vector<std::array<unsigned int, 3>> const& indices);

  /** Compute inertia tensor for given geometry and given density.
   */
  void compute_inertia_tensor(F density);

  /** Return mass of body. */
  F getMass() const { return m_mass; }
  /** Return volume of body. */
  F getVolume() const { return m_volume; }
  /** Return inertia tensor of body. */
  Matrix3r const& getInertia() const { return m_theta; }
  /** Return center of mass. */
  Vector3r const& getCenterOfMass() const { return m_r; }

 private:
  void volume_integrals();
  void face_integrals(unsigned int i);

  /** Compute various integrations over projection of face.
   */
  void projection_integrals(unsigned int i);

  std::vector<Vector3r> m_face_normals;
  std::vector<F> m_weights;
  unsigned int m_nVertices;
  unsigned int m_nFaces;
  std::vector<Vector3r> m_vertices;
  std::vector<std::array<unsigned int, 3>> m_indices;

  F m_mass, m_volume;
  Vector3r m_r;
  Vector3r m_x;
  Matrix3r m_theta;
};
}  // namespace dg
}  // namespace alluvion

#endif
