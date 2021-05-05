#ifndef ALLUVION_DG_MESH_DISTANCE_HPP
#define ALLUVION_DG_MESH_DISTANCE_HPP

#include "alluvion/dg/common.hpp"
#include "alluvion/dg/triangle_mesh.hpp"

namespace std {
template <>
struct hash<alluvion::dg::Vector3r> {
  std::size_t operator()(alluvion::dg::Vector3r const& x) const {
    std::size_t seed = 0;
    std::hash<alluvion::F> hasher;
    seed ^= hasher(x[0]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hasher(x[1]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hasher(x[2]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

template <>
struct less<alluvion::dg::Vector3r> {
  bool operator()(alluvion::dg::Vector3r const& left,
                  alluvion::dg::Vector3r const& right) const {
    for (auto i = 0u; i < 3u; ++i) {
      if (left(i) < right(i))
        return true;
      else if (left(i) > right(i))
        return false;
    }
    return false;
  }
};
}  // namespace std
#include <Eigen/Dense>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "alluvion/dg/bounding_sphere_hierarchy.hpp"
#include "alluvion/dg/distance.hpp"
#include "alluvion/dg/lru_cache.hpp"

namespace alluvion {
namespace dg {

enum class NearestEntity;
class TriangleMesh;
class Halfedge;
class MeshDistance : public Distance {
  struct Candidate {
    bool operator<(Candidate const& other) const { return b < other.b; }
    unsigned int node_index;
    F b, w;
  };

 public:
  MeshDistance(TriangleMesh const& mesh, bool precompute_normals = true);

  // Returns the shortest unsigned distance from a given point x to
  // the stored mesh.
  // Thread-safe function.
  F distance(dg::Vector3r const& x, dg::Vector3r* nearest_point = nullptr,
             unsigned int* nearest_face = nullptr,
             NearestEntity* ne = nullptr) const;

  // Requires a closed two-manifold mesh as input data.
  // Thread-safe function.
  F signedDistance(dg::Vector3r const& x) const override;
  F signedDistanceCached(dg::Vector3r const& x) const;

  F unsignedDistance(dg::Vector3r const& x) const;
  F unsignedDistanceCached(dg::Vector3r const& x) const;

 private:
  dg::Vector3r vertex_normal(unsigned int v) const;
  dg::Vector3r edge_normal(Halfedge const& h) const;
  dg::Vector3r face_normal(unsigned int f) const;

  void callback(unsigned int node_index, TriangleMeshBSH const& bsh,
                dg::Vector3r const& x, F& dist) const;

  bool predicate(unsigned int node_index, TriangleMeshBSH const& bsh,
                 dg::Vector3r const& x, F& dist) const;

 private:
  TriangleMesh m_mesh;
  TriangleMeshBSH m_bsh;

  using FunctionValueCache = LRUCache<dg::Vector3r, F>;
  mutable std::vector<TriangleMeshBSH::TraversalQueue> m_queues;
  mutable std::vector<unsigned int> m_nearest_face;
  mutable std::vector<FunctionValueCache> m_cache;
  mutable std::vector<FunctionValueCache> m_ucache;

  std::vector<dg::Vector3r> m_face_normals;
  std::vector<dg::Vector3r> m_vertex_normals;
  bool m_precomputed_normals;
};

}  // namespace dg
}  // namespace alluvion

#endif
