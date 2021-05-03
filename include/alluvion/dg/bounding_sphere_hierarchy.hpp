#ifndef ALLUVION_DG_BOUNDING_SPHERE_HIERARCHY_HPP
#define ALLUVION_DG_BOUNDING_SPHERE_HIERARCHY_HPP

#include "alluvion/dg/bounding_sphere.hpp"
#include "alluvion/dg/kd_tree.hpp"

namespace alluvion {
namespace dg {

class TriangleMeshBSH : public KDTree<BoundingSphere> {
 public:
  using super = KDTree<BoundingSphere>;

  TriangleMeshBSH(std::vector<Vector3r> const& vertices,
                  std::vector<std::array<unsigned int, 3>> const& faces);

  Vector3r const& entityPosition(unsigned int i) const final;
  void computeHull(unsigned int b, unsigned int n,
                   BoundingSphere& hull) const final;

 private:
  std::vector<Vector3r> const& m_vertices;
  std::vector<std::array<unsigned int, 3>> const& m_faces;

  std::vector<Vector3r> m_tri_centers;
};

}  // namespace dg
}  // namespace alluvion

#endif
