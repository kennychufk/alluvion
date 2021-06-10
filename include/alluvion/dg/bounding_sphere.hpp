#ifndef ALLUVION_DG_BOUNDING_SPHERE_HPP
#define ALLUVION_DG_BOUNDING_SPHERE_HPP

#include <Eigen/Core>
#include <vector>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {
/**
 * \brief Computes smallest enclosing spheres of pointsets using Welzl's
 * algorithm \Author: Tassilo Kugelstadt
 */
template <typename TF>
class BoundingSphere {
 public:
  /**
   * \brief default constructor sets the center and radius to zero.
   */
  BoundingSphere() : m_x(Vector3r<TF>::Zero()), m_r(0.0) {}

  /**
   * \brief constructor which sets the center and radius
   *
   * \param x	3d coordinates of the center point
   * \param r radius of the sphere
   */
  BoundingSphere(const Vector3r<TF>& x, TF r) : m_x(x), m_r(r) {}

  /**
   * \brief	constructs a sphere for one point (with radius 0)
   *
   * \param a	3d coordinates of point a
   */
  BoundingSphere(const Vector3r<TF>& a) {
    m_x = a;
    m_r = 0.0;
  }

  /**
   * \brief	constructs the smallest enclosing sphere for two points
   *
   * \param a 3d coordinates of point a
   * \param b 3d coordinates of point b
   */
  BoundingSphere(const Vector3r<TF>& a, const Vector3r<TF>& b) {
    const Vector3r<TF> ba = b - a;

    m_x = (a + b) * 0.5;
    m_r = 0.5 * ba.norm();
  }

  /**
   * \brief	constructs the smallest enclosing sphere for three points
   *
   * \param a 3d coordinates of point a
   * \param b 3d coordinates of point b
   * \param c 3d coordinates of point c
   */
  BoundingSphere(const Vector3r<TF>& a, const Vector3r<TF>& b,
                 const Vector3r<TF>& c) {
    const Vector3r<TF> ba = b - a;
    const Vector3r<TF> ca = c - a;
    const Vector3r<TF> baxca = ba.cross(ca);
    Vector3r<TF> r;
    Matrix3r<TF> T;
    T << ba[0], ba[1], ba[2], ca[0], ca[1], ca[2], baxca[0], baxca[1], baxca[2];

    r[0] = 0.5 * ba.squaredNorm();
    r[1] = 0.5 * ca.squaredNorm();
    r[2] = 0.0;

    m_x = T.inverse() * r;
    m_r = m_x.norm();
    m_x += a;
  }

  /**
   * \brief constructs the smallest enclosing sphere for four points
   *
   * \param a 3d coordinates of point a
   * \param b 3d coordinates of point b
   * \param c 3d coordinates of point c
   * \param d 3d coordinates of point d
   */
  BoundingSphere(const Vector3r<TF>& a, const Vector3r<TF>& b,
                 const Vector3r<TF>& c, const Vector3r<TF>& d) {
    const Vector3r<TF> ba = b - a;
    const Vector3r<TF> ca = c - a;
    const Vector3r<TF> da = d - a;
    Vector3r<TF> r;
    Matrix3r<TF> T;
    T << ba[0], ba[1], ba[2], ca[0], ca[1], ca[2], da[0], da[1], da[2];

    r[0] = 0.5 * ba.squaredNorm();
    r[1] = 0.5 * ca.squaredNorm();
    r[2] = 0.5 * da.squaredNorm();
    m_x = T.inverse() * r;
    m_r = m_x.norm();
    m_x += a;
  }

  /**
   * \brief	constructs the smallest enclosing sphere a given pointset
   *
   * \param p vertices of the points
   */
  BoundingSphere(const std::vector<Vector3r<TF>>& p) {
    m_r = 0;
    m_x.setZero();
    setPoints(p);
  }

  /**
   * \brief	Getter for the center of the sphere
   *
   * \return	const reference of the sphere center
   */
  Vector3r<TF> const& x() const { return m_x; }

  /**
   * \brief	Access function for center of the sphere
   *
   * \return	reference of the sphere center
   */
  Vector3r<TF>& x() { return m_x; }

  /**
   * \brief	Getter for the radius
   *
   * \return	Radius of the sphere
   */
  TF r() const { return m_r; }

  /**
   * \brief	Access function for the radius
   *
   * \return	Reference to the radius of the sphere
   */
  TF& r() { return m_r; }

  /**
   * \brief	constructs the smallest enclosing sphere a given pointset
   *
   * \param p vertices of the points
   */
  void setPoints(const std::vector<Vector3r<TF>>& p) {
    // remove duplicates
    std::vector<Vector3r<TF>> v(p);
    std::sort(v.begin(), v.end(),
              [](const Vector3r<TF>& a, const Vector3r<TF>& b) {
                if (a[0] < b[0]) return true;
                if (a[0] > b[0]) return false;
                if (a[1] < b[1]) return true;
                if (a[1] > b[1]) return false;
                return (a[2] < b[2]);
              });
    v.erase(std::unique(
                v.begin(), v.end(),
                [](Vector3r<TF>& a, Vector3r<TF>& b) { return a.isApprox(b); }),
            v.end());

    Vector3r<TF> d;
    const int n = int(v.size());

    // generate random permutation of the points and perturb the points by
    // epsilon to avoid corner cases
    const TF epsilon = 1.0e-6;
    for (int i = n - 1; i > 0; i--) {
      const Vector3r<TF> epsilon_vec = epsilon * Vector3r<TF>::Random();
      const int j = static_cast<int>(floor(i * TF(rand()) / RAND_MAX));
      d = v[i] + epsilon_vec;
      v[i] = v[j] - epsilon_vec;
      v[j] = d;
    }

    BoundingSphere S = BoundingSphere(v[0], v[1]);

    for (int i = 2; i < n; i++) {
      // SES0
      d = v[i] - S.x();
      if (d.squaredNorm() > S.r() * S.r()) S = ses1(i, v, v[i]);
    }

    m_x = S.m_x;
    m_r = S.m_r + epsilon;  // add epsilon to make sure that all non-perturbed
                            // points are inside the sphere
  }

  /**
   * \brief		intersection test for two spheres
   *
   * \param other other sphere to be tested for intersection
   * \return		returns true when this sphere and the other sphere are
   * intersecting
   */
  bool overlaps(BoundingSphere const& other) const {
    const TF rr = m_r + other.m_r;
    return (m_x - other.m_x).squaredNorm() < rr * rr;
  }

  /**
   * \brief		tests whether the given sphere other is contained in the
   * sphere
   *
   * \param		other bounding sphere
   * \return		returns true when the other is contained in this sphere
   * or vice versa
   */
  bool contains(BoundingSphere const& other) const {
    const TF rr = r() - other.r();
    return (x() - other.x()).squaredNorm() < rr * rr;
  }

  /**
   * \brief		tests whether the given point other is contained in the
   * sphere
   *
   * \param		other 3d coordinates of a point
   * \return		returns true when the point is contained in the sphere
   */
  bool contains(Vector3r<TF> const& other) const {
    return (x() - other).squaredNorm() < m_r * m_r;
  }

 private:
  /**
   * \brief		constructs the smallest enclosing sphere for n points
   * with the points q1, q2, and q3 on the surface of the sphere
   *
   * \param n		number of points
   * \param p		vertices of the points
   * \param q1	3d coordinates of a point on the surface
   * \param q2	3d coordinates of a second point on the surface
   * \param q3	3d coordinates of a third point on the surface
   * \return		smallest enclosing sphere
   */
  BoundingSphere ses3(int n, std::vector<Vector3r<TF>>& p, Vector3r<TF>& q1,
                      Vector3r<TF>& q2, Vector3r<TF>& q3) {
    BoundingSphere S(q1, q2, q3);

    for (int i = 0; i < n; i++) {
      Vector3r<TF> d = p[i] - S.x();
      if (d.squaredNorm() > S.r() * S.r()) S = BoundingSphere(q1, q2, q3, p[i]);
    }
    return S;
  }

  /**
   * \brief		constructs the smallest enclosing sphere for n points
   * with the points q1 and q2 on the surface of the sphere
   *
   * \param n		number of points
   * \param p		vertices of the points
   * \param q1	3d coordinates of a point on the surface
   * \param q2	3d coordinates of a second point on the surface
   * \return		smallest enclosing sphere
   */
  BoundingSphere ses2(int n, std::vector<Vector3r<TF>>& p, Vector3r<TF>& q1,
                      Vector3r<TF>& q2) {
    BoundingSphere S(q1, q2);

    for (int i = 0; i < n; i++) {
      Vector3r<TF> d = p[i] - S.x();
      if (d.squaredNorm() > S.r() * S.r()) S = ses3(i, p, q1, q2, p[i]);
    }
    return S;
  }
  /**
   * \brief		constructs the smallest enclosing sphere for n points
   * with the point q1 on the surface of the sphere
   *
   * \param n		number of points
   * \param p		vertices of the points
   * \param q1	3d coordinates of a point on the surface
   * \return		smallest enclosing sphere
   */
  BoundingSphere ses1(int n, std::vector<Vector3r<TF>>& p, Vector3r<TF>& q1) {
    BoundingSphere S(p[0], q1);

    for (int i = 1; i < n; i++) {
      Vector3r<TF> d = p[i] - S.x();
      if (d.squaredNorm() > S.r() * S.r()) S = ses2(i, p, q1, p[i]);
    }
    return S;
  }

  Vector3r<TF> m_x;
  TF m_r;
};

}  // namespace dg
}  // namespace alluvion

#endif
