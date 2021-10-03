#ifndef ALLUVION_DG_TRIANGLE_MESH_HPP
#define ALLUVION_DG_TRIANGLE_MESH_HPP

#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "alluvion/dg/common.hpp"
#include "alluvion/dg/entity_containers.hpp"
#include "alluvion/dg/halfedge.hpp"

namespace alluvion {
namespace dg {

template <typename TU>
struct HalfedgeHasher {
  HalfedgeHasher(std::vector<std::array<TU, 3>> const& faces_)
      : faces(&faces_) {}

  std::size_t operator()(Halfedge<TU> const& he) const {
    TU f = he.face();
    TU e = he.edge();
    std::array<TU, 2> v = {(*faces)[f][e], (*faces)[f][(e + 1) % 3]};
    if (v[0] > v[1]) std::swap(v[0], v[1]);

    std::size_t seed(0);
    hash_combine(seed, v[0]);
    hash_combine(seed, v[1]);
    return seed;
  }
  static inline void hash_combine(std::size_t& seed, const TU& v) {
    std::hash<TU> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  std::vector<std::array<TU, 3>> const* faces;
};

template <typename TU>
struct HalfedgeEqualTo {
  HalfedgeEqualTo(std::vector<std::array<TU, 3>> const& faces_)
      : faces(&faces_) {}

  bool operator()(Halfedge<TU> const& a, Halfedge<TU> const& b) const {
    TU fa = a.face();
    TU ea = a.edge();
    std::array<TU, 2> va = {(*faces)[fa][ea], (*faces)[fa][(ea + 1) % 3]};

    TU fb = b.face();
    TU eb = b.edge();
    std::array<TU, 2> vb = {(*faces)[fb][eb], (*faces)[fb][(eb + 1) % 3]};

    return va[0] == vb[1] && va[1] == vb[0];
  }

  std::vector<std::array<TU, 3>> const* faces;
};

template <typename TF>
class TriangleMesh {
 public:
  using FaceSet =
      std::unordered_set<Halfedge<unsigned int>, HalfedgeHasher<unsigned int>,
                         HalfedgeEqualTo<unsigned int>>;
  TriangleMesh() {}

  TriangleMesh(std::vector<dg::Vector3r<TF>> const& vertices,
               std::vector<std::array<unsigned int, 3>> const& faces)
      : m_faces(faces),
        m_e2e(3 * faces.size()),
        m_vertices(vertices),
        m_v2e(vertices.size()) {
    construct();
  }

  TriangleMesh(TF const* vertices, unsigned int const* faces, std::size_t nv,
               std::size_t nf)
      : m_faces(nf), m_vertices(nv), m_e2e(3 * nf), m_v2e(nv) {
    std::copy(vertices, vertices + 3 * nv, m_vertices[0].data());
    std::copy(faces, faces + 3 * nf, m_faces[0].data());
    construct();
  }

  TriangleMesh(std::string const& path) {
    std::ifstream in(path, std::ios::in);
    if (!in) {
      std::cerr << "Cannot open " << path << std::endl;
      return;
    }

    std::string line;
    while (getline(in, line)) {
      if (line.substr(0, 2) == "v ") {
        std::istringstream s(line.substr(2));
        Vector3r<TF> v;
        s >> v.x();
        s >> v.y();
        s >> v.z();
        m_vertices.push_back(v);
      } else if (line.substr(0, 2) == "f ") {
        std::istringstream s(line.substr(2));
        std::array<unsigned int, 3> f;
        for (unsigned int j(0); j < 3; ++j) {
          std::string buf;
          s >> buf;
          buf = buf.substr(0, buf.find_first_of('/'));
          f[j] = std::stoi(buf) - 1;
        }
        m_faces.push_back(f);
      } else if (line[0] == '#') { /* ignoring this line */
      } else {                     /* ignoring this line */
      }
    }

    construct();
  }

  void exportOBJ(std::string const& filename) const {
    auto outfile = std::ofstream(filename.c_str());
    auto str_stream = std::stringstream(std::stringstream::in);

    outfile << "g default" << std::endl;
    for (auto const& pos : m_vertices) {
      outfile << "v " << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
    }

    for (auto const& f : m_faces) {
      outfile << "f";
      for (auto v : f) outfile << " " << v + 1;
      outfile << std::endl;
    }

    outfile.close();
  }

  // Halfedge modifiers.
  unsigned int source(Halfedge<unsigned int> const h) const {
    if (h.isBoundary()) return target(opposite(h));
    return m_faces[h.face()][h.edge()];
  }
  unsigned int target(Halfedge<unsigned int> const h) const {
    if (h.isBoundary()) return source(opposite(h));
    return source(h.next());
  }
  Halfedge<unsigned int> opposite(Halfedge<unsigned int> const h) const {
    if (h.isBoundary()) return m_b2e[h.face()];
    return m_e2e[h.face()][h.edge()];
  }

  // Container getters.
  FaceContainer<TF> faces() { return FaceContainer<TF>(this); }
  FaceConstContainer<TF> faces() const { return FaceConstContainer<TF>(this); }
  IncidentFaceContainer<TF> incident_faces(unsigned int v) const {
    return IncidentFaceContainer<TF>(v, this);
  }
  VertexContainer<TF> vertices() { return VertexContainer<TF>(this); }
  VertexConstContainer<TF> vertices() const {
    return VertexConstContainer<TF>(this);
  }

  // Entity size getters.
  std::size_t nFaces() const { return m_faces.size(); }
  std::size_t nVertices() const { return m_v2e.size(); }
  std::size_t nBorderEdges() const { return m_b2e.size(); }

  // Entity getters.
  unsigned int const& faceVertex(unsigned int f, unsigned int i) const {
    assert(i < 3);
    assert(f < m_faces.size());
    return m_faces[f][i];
  }
  unsigned int& faceVertex(unsigned int f, unsigned int i) {
    assert(i < 3);
    assert(f < m_faces.size());
    return m_faces[f][i];
  }

  dg::Vector3r<TF> const& vertex(unsigned int i) const { return m_vertices[i]; }
  dg::Vector3r<TF>& vertex(unsigned int i) { return m_vertices[i]; }
  std::array<unsigned int, 3> const& face(unsigned int i) const {
    return m_faces[i];
  }
  std::array<unsigned int, 3>& face(unsigned int i) { return m_faces[i]; }
  Halfedge<unsigned int> incident_halfedge(unsigned int v) const {
    return m_v2e[v];
  }

  // Data getters.
  std::vector<dg::Vector3r<TF>> const& vertex_data() const {
    return m_vertices;
  }
  std::vector<dg::Vector3r<TF>>& vertex_data() { return m_vertices; }
  std::vector<std::array<unsigned int, 3>> const& face_data() const {
    return m_faces;
  }
  std::vector<std::array<unsigned int, 3>>& face_data() { return m_faces; }

  dg::Vector3r<TF> computeFaceNormal(unsigned int f) const {
    Vector3r<TF> const& x0 = vertex(faceVertex(f, 0));
    Vector3r<TF> const& x1 = vertex(faceVertex(f, 1));
    Vector3r<TF> const& x2 = vertex(faceVertex(f, 2));

    return (x1 - x0).cross(x2 - x0).normalized();
  }

  void construct() {
    m_e2e.resize(3 * m_faces.size());
    m_v2e.resize(m_vertices.size());

    // Build adjacencies for mesh faces.
    FaceSet face_set((m_faces.size() * 3) / 2, HalfedgeHasher(m_faces),
                     HalfedgeEqualTo(m_faces));
    for (unsigned int i(0); i < m_faces.size(); ++i)
      for (unsigned char j(0); j < 3; ++j) {
        Halfedge<unsigned int> he(i, j);
        auto ret = face_set.insert(he);
        if (!ret.second) {
          m_e2e[he.face()][he.edge()] = *(ret.first);
          m_e2e[ret.first->face()][ret.first->edge()] = he;

          face_set.erase(ret.first);
        }

        m_v2e[m_faces[i][j]] = he;
      }

    m_b2e.reserve(face_set.size());

    for (Halfedge<unsigned int> const he : face_set) {
      m_b2e.push_back(he);
      Halfedge<unsigned int> b(static_cast<unsigned int>(m_b2e.size()) - 1u, 3);
      m_e2e[he.face()][he.edge()] = b;
      m_v2e[target(he)] = b;

      assert(source(b) == target(he));
    }

#ifdef _DEBUG
    for (unsigned int i(0); i < nFaces(); ++i) {
      Halfedge<unsigned int> h(i, 0);
      for (unsigned int j(0); j < 3; ++j) {
        assert(faceVertex(i, j) == source(h));
        h = h.next();
      }
    }
#endif

    if (!m_b2e.empty()) {
      std::cout << std::endl << "WARNING: Mesh not closed!" << std::endl;
    }
  }

  std::vector<dg::Vector3r<TF>> m_vertices;
  std::vector<std::array<unsigned int, 3>> m_faces;
  std::vector<std::array<Halfedge<unsigned int>, 3>> m_e2e;
  std::vector<Halfedge<unsigned int>> m_v2e;
  std::vector<Halfedge<unsigned int>> m_b2e;
};
}  // namespace dg
}  // namespace alluvion

#endif
