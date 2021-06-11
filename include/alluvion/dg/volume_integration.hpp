#ifndef ALLUVION_DG_VOLUME_INTEGRATION_HPP
#define ALLUVION_DG_VOLUME_INTEGRATION_HPP

#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {
template <typename TF>
class VolumeIntegration {
 private:
  int A;
  int B;
  int C;

  // projection integrals
  TF P1, Pa, Pb, Paa, Pab, Pbb, Paaa, Paab, Pabb, Pbbb;
  // face integrals
  TF Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca;
  // volume integrals
  TF T0;
  TF T1[3];
  TF T2[3];
  TF TP[3];

 public:
  VolumeIntegration(std::vector<Vector3r<TF>> const& vertices,
                    std::vector<std::array<unsigned int, 3>> const& indices)
      : m_nVertices(vertices.size()),
        m_nFaces(indices.size()),
        m_indices(indices),
        m_face_normals(indices.size()),
        m_weights(indices.size()) {
    // compute center of mass
    m_x.setZero();
    for (unsigned int i(0); i < m_nVertices; ++i) m_x += vertices[i];
    m_x /= (TF)m_nVertices;

    m_vertices.resize(vertices.size());
    for (unsigned int i(0); i < m_nVertices; ++i)
      m_vertices[i] = vertices[i] - m_x;

    for (unsigned int i(0); i < m_nFaces; ++i) {
      const Vector3r<TF>& a = m_vertices[m_indices[i][0]];
      const Vector3r<TF>& b = m_vertices[m_indices[i][1]];
      const Vector3r<TF>& c = m_vertices[m_indices[i][2]];

      const Vector3r<TF> d1 = b - a;
      const Vector3r<TF> d2 = c - a;

      m_face_normals[i] = d1.cross(d2);
      if (m_face_normals[i].isZero(1.e-10))
        m_face_normals[i].setZero();
      else
        m_face_normals[i].normalize();

      m_weights[i] = -m_face_normals[i].dot(a);
    }
  }

  /** Compute inertia tensor for given geometry and given density.
   */
  void compute_inertia_tensor(TF density) {
    volume_integrals();
    m_volume = static_cast<TF>(T0);

    m_mass = static_cast<TF>(density * T0);

    /* compute center of mass */
    m_r[0] = static_cast<TF>(T1[0] / T0);
    m_r[1] = static_cast<TF>(T1[1] / T0);
    m_r[2] = static_cast<TF>(T1[2] / T0);

    /* compute inertia tensor */
    m_theta(0, 0) = static_cast<TF>(density * (T2[1] + T2[2]));
    m_theta(1, 1) = static_cast<TF>(density * (T2[2] + T2[0]));
    m_theta(2, 2) = static_cast<TF>(density * (T2[0] + T2[1]));
    m_theta(0, 1) = m_theta(1, 0) = -density * static_cast<TF>(TP[0]);
    m_theta(1, 2) = m_theta(2, 1) = -density * static_cast<TF>(TP[1]);
    m_theta(2, 0) = m_theta(0, 2) = -density * static_cast<TF>(TP[2]);

    /* translate inertia tensor to center of mass */
    m_theta(0, 0) -= m_mass * (m_r[1] * m_r[1] + m_r[2] * m_r[2]);
    m_theta(1, 1) -= m_mass * (m_r[2] * m_r[2] + m_r[0] * m_r[0]);
    m_theta(2, 2) -= m_mass * (m_r[0] * m_r[0] + m_r[1] * m_r[1]);
    m_theta(0, 1) = m_theta(1, 0) += m_mass * m_r[0] * m_r[1];
    m_theta(1, 2) = m_theta(2, 1) += m_mass * m_r[1] * m_r[2];
    m_theta(2, 0) = m_theta(0, 2) += m_mass * m_r[2] * m_r[0];

    m_r += m_x;
  }

  /** Return mass of body. */
  TF getMass() const { return m_mass; }
  /** Return volume of body. */
  TF getVolume() const { return m_volume; }
  /** Return inertia tensor of body. */
  Matrix3r<TF> const& getInertia() const { return m_theta; }
  /** Return center of mass. */
  Vector3r<TF> const& getCenterOfMass() const { return m_r; }

 private:
  void volume_integrals() {
    TF nx, ny, nz;

    T0 = T1[0] = T1[1] = T1[2] = T2[0] = T2[1] = T2[2] = TP[0] = TP[1] = TP[2] =
        0;

    for (unsigned int i(0); i < m_nFaces; ++i) {
      Vector3r<TF> const& n = m_face_normals[i];
      nx = std::abs(n[0]);
      ny = std::abs(n[1]);
      nz = std::abs(n[2]);
      if (nx > ny && nx > nz)
        C = 0;
      else
        C = (ny > nz) ? 1 : 2;
      A = (C + 1) % 3;
      B = (A + 1) % 3;

      face_integrals(i);

      T0 += n[0] * ((A == 0) ? Fa : ((B == 0) ? Fb : Fc));

      T1[A] += n[A] * Faa;
      T1[B] += n[B] * Fbb;
      T1[C] += n[C] * Fcc;
      T2[A] += n[A] * Faaa;
      T2[B] += n[B] * Fbbb;
      T2[C] += n[C] * Fccc;
      TP[A] += n[A] * Faab;
      TP[B] += n[B] * Fbbc;
      TP[C] += n[C] * Fcca;
    }

    T1[0] /= 2;
    T1[1] /= 2;
    T1[2] /= 2;
    T2[0] /= 3;
    T2[1] /= 3;
    T2[2] /= 3;
    TP[0] /= 2;
    TP[1] /= 2;
    TP[2] /= 2;
  }
  void face_integrals(unsigned int i) {
    TF w;
    Vector3r<TF> n;
    TF k1, k2, k3, k4;

    projection_integrals(f);

    w = m_weights[f];
    n = m_face_normals[f];
    k1 = (n[C] == 0) ? 0 : 1 / n[C];
    k2 = k1 * k1;
    k3 = k2 * k1;
    k4 = k3 * k1;

    Fa = k1 * Pa;
    Fb = k1 * Pb;
    Fc = -k2 * (n[A] * Pa + n[B] * Pb + w * P1);

    Faa = k1 * Paa;
    Fbb = k1 * Pbb;
    Fcc = k3 * (SQR(n[A]) * Paa + 2 * n[A] * n[B] * Pab + SQR(n[B]) * Pbb +
                w * (2 * (n[A] * Pa + n[B] * Pb) + w * P1));

    Faaa = k1 * Paaa;
    Fbbb = k1 * Pbbb;
    Fccc =
        -k4 *
        (CUBE(n[A]) * Paaa + 3 * SQR(n[A]) * n[B] * Paab +
         3 * n[A] * SQR(n[B]) * Pabb + CUBE(n[B]) * Pbbb +
         3 * w * (SQR(n[A]) * Paa + 2 * n[A] * n[B] * Pab + SQR(n[B]) * Pbb) +
         w * w * (3 * (n[A] * Pa + n[B] * Pb) + w * P1));

    Faab = k1 * Paab;
    Fbbc = -k2 * (n[A] * Pabb + n[B] * Pbbb + w * Pbb);
    Fcca = k3 * (SQR(n[A]) * Paaa + 2 * n[A] * n[B] * Paab + SQR(n[B]) * Pabb +
                 w * (2 * (n[A] * Paa + n[B] * Pab) + w * Pa));
  }

  /** Compute various integrations over projection of face.
   */
  void projection_integrals(unsigned int i) {
    TF a0, a1, da;
    TF b0, b1, db;
    TF a0_2, a0_3, a0_4, b0_2, b0_3, b0_4;
    TF a1_2, a1_3, b1_2, b1_3;
    TF C1, Ca, Caa, Caaa, Cb, Cbb, Cbbb;
    TF Cab, Kab, Caab, Kaab, Cabb, Kabb;

    P1 = Pa = Pb = Paa = Pab = Pbb = Paaa = Paab = Pabb = Pbbb = 0.0;

    for (int i = 0; i < 3; i++) {
      a0 = m_vertices[m_indices[f][i]][A];
      b0 = m_vertices[m_indices[f][i]][B];
      a1 = m_vertices[m_indices[f][(i + 1) % 3]][A];
      b1 = m_vertices[m_indices[f][(i + 1) % 3]][B];

      da = a1 - a0;
      db = b1 - b0;
      a0_2 = a0 * a0;
      a0_3 = a0_2 * a0;
      a0_4 = a0_3 * a0;
      b0_2 = b0 * b0;
      b0_3 = b0_2 * b0;
      b0_4 = b0_3 * b0;
      a1_2 = a1 * a1;
      a1_3 = a1_2 * a1;
      b1_2 = b1 * b1;
      b1_3 = b1_2 * b1;

      C1 = a1 + a0;
      Ca = a1 * C1 + a0_2;
      Caa = a1 * Ca + a0_3;
      Caaa = a1 * Caa + a0_4;
      Cb = b1 * (b1 + b0) + b0_2;
      Cbb = b1 * Cb + b0_3;
      Cbbb = b1 * Cbb + b0_4;
      Cab = 3 * a1_2 + 2 * a1 * a0 + a0_2;
      Kab = a1_2 + 2 * a1 * a0 + 3 * a0_2;
      Caab = a0 * Cab + 4 * a1_3;
      Kaab = a1 * Kab + 4 * a0_3;
      Cabb = 4 * b1_3 + 3 * b1_2 * b0 + 2 * b1 * b0_2 + b0_3;
      Kabb = b1_3 + 2 * b1_2 * b0 + 3 * b1 * b0_2 + 4 * b0_3;

      P1 += db * C1;
      Pa += db * Ca;
      Paa += db * Caa;
      Paaa += db * Caaa;
      Pb += da * Cb;
      Pbb += da * Cbb;
      Pbbb += da * Cbbb;
      Pab += db * (b1 * Cab + b0 * Kab);
      Paab += db * (b1 * Caab + b0 * Kaab);
      Pabb += da * (a1 * Cabb + a0 * Kabb);
    }

    P1 /= 2.0;
    Pa /= 6.0;
    Paa /= 12.0;
    Paaa /= 20.0;
    Pb /= -6.0;
    Pbb /= -12.0;
    Pbbb /= -20.0;
    Pab /= 24.0;
    Paab /= 60.0;
    Pabb /= -60.0;
  }

  std::vector<Vector3r<TF>> m_face_normals;
  std::vector<TF> m_weights;
  unsigned int m_nVertices;
  unsigned int m_nFaces;
  std::vector<Vector3r<TF>> m_vertices;
  std::vector<std::array<unsigned int, 3>> m_indices;

  TF m_mass, m_volume;
  Vector3r<TF> m_r;
  Vector3r<TF> m_x;
  Matrix3r<TF> m_theta;
};
}  // namespace dg
}  // namespace alluvion

#endif
