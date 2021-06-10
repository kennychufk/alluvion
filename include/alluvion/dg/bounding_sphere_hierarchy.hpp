#ifndef ALLUVION_DG_BOUNDING_SPHERE_HIERARCHY_HPP
#define ALLUVION_DG_BOUNDING_SPHERE_HIERARCHY_HPP

#include "alluvion/dg/bounding_sphere.hpp"
#include "alluvion/dg/kd_tree.hpp"

namespace alluvion {
namespace dg {

template <typename TF>
class TriangleMeshBSH : public KDTree<BoundingSphere<TF>, TF> {
 public:
  using super = KDTree<BoundingSphere<TF>, TF>;

  TriangleMeshBSH(std::vector<Vector3r<TF>> const& vertices,
                  std::vector<std::array<unsigned int, 3>> const& faces)
      : super(faces.size()),
        m_faces(faces),
        m_vertices(vertices),
        m_tri_centers(faces.size()) {
    std::transform(
        m_faces.begin(), m_faces.end(), m_tri_centers.begin(),
        [&](std::array<unsigned int, 3> const& f) {
          return 1.0 / 3.0 *
                 (m_vertices[f[0]] + m_vertices[f[1]] + m_vertices[f[2]]);
        });
  }

  Vector3r<TF> const& entityPosition(unsigned int i) const final {
    return m_tri_centers[i];
  }
  void computeHull(unsigned int b, unsigned int n,
                   BoundingSphere<TF>& hull) const final {
    auto vertices_subset = std::vector<Vector3r<TF>>(3 * n);
    for (unsigned int i(0); i < n; ++i) {
      auto const& f = m_faces[super::m_lst[b + i]];
      {
        vertices_subset[3 * i + 0] = m_vertices[f[0]];
        vertices_subset[3 * i + 1] = m_vertices[f[1]];
        vertices_subset[3 * i + 2] = m_vertices[f[2]];
      }
    }

    const BoundingSphere<TF> s(vertices_subset);

    hull.x() = s.x();
    hull.r() = s.r();
  }

 private:
  std::vector<Vector3r<TF>> const& m_vertices;
  std::vector<std::array<unsigned int, 3>> const& m_faces;

  std::vector<Vector3r<TF>> m_tri_centers;
};

}  // namespace dg
}  // namespace alluvion

#endif
