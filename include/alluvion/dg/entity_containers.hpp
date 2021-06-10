#ifndef ALLUVION_DG_ENTITY_CONTAINERS_HPP
#define ALLUVION_DG_ENTITY_CONTAINERS_HPP

#include "alluvion/dg/entity_iterators.hpp"

namespace alluvion {
namespace dg {

template <typename TF>
class TriangleMesh;

template <typename TF>
class FaceContainer {
 public:
  FaceIterator<TF> begin() const { return FaceIterator<TF>(0, m_mesh); }
  FaceIterator<TF> end() const {
    return FaceIterator<TF>(static_cast<unsigned int>(m_mesh->nFaces()),
                            m_mesh);
  }

 private:
  friend class TriangleMesh<TF>;
  FaceContainer(TriangleMesh<TF>* mesh) : m_mesh(mesh) {}

  TriangleMesh<TF>* m_mesh;
};

template <typename TF>
class FaceConstContainer {
 public:
  FaceConstIterator<TF> begin() const {
    return FaceConstIterator<TF>(0, m_mesh);
  }
  FaceConstIterator<TF> end() const {
    return FaceConstIterator<TF>(static_cast<unsigned int>(m_mesh->nFaces()),
                                 m_mesh);
  }

 private:
  friend class TriangleMesh<TF>;
  FaceConstContainer(TriangleMesh<TF> const* mesh) : m_mesh(mesh) {}

  TriangleMesh<TF> const* m_mesh;
};

template <typename TF>
class IncidentFaceContainer {
 public:
  IncidentFaceIterator<TF> begin() const {
    return IncidentFaceIterator<TF>(m_v, m_mesh);
  }
  IncidentFaceIterator<TF> end() const { return IncidentFaceIterator<TF>(); }

 private:
  friend class TriangleMesh<TF>;
  IncidentFaceContainer(unsigned int v, TriangleMesh<TF> const* mesh)
      : m_v(v), m_mesh(mesh) {}

  TriangleMesh<TF> const* m_mesh;
  unsigned int m_v;
};

template <typename TF>
class VertexContainer {
 public:
  VertexIterator<TF> begin() const { return VertexIterator<TF>(0, m_mesh); }
  VertexIterator<TF> end() const {
    return VertexIterator<TF>(static_cast<unsigned int>(m_mesh->nVertices()),
                              m_mesh);
  }

 private:
  friend class TriangleMesh<TF>;
  VertexContainer(TriangleMesh<TF>* mesh) : m_mesh(mesh) {}

  TriangleMesh<TF>* m_mesh;
};

template <typename TF>
class VertexConstContainer {
 public:
  VertexConstIterator<TF> begin() const {
    return VertexConstIterator<TF>(0, m_mesh);
  }
  VertexConstIterator<TF> end() const {
    return VertexConstIterator<TF>(
        static_cast<unsigned int>(m_mesh->nVertices()), m_mesh);
  }

 private:
  friend class TriangleMesh<TF>;
  VertexConstContainer(TriangleMesh<TF> const* mesh) : m_mesh(mesh) {}

  TriangleMesh<TF> const* m_mesh;
};
}  // namespace dg
}  // namespace alluvion

#endif
