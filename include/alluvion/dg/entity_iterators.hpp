#ifndef ALLUVION_DG_ENTITY_ITERATORS_HPP
#define ALLUVION_DG_ENTITY_ITERATORS_HPP

#include <Eigen/Core>
#include <array>
#include <iterator>

#include "alluvion/dg/common.hpp"
#include "alluvion/dg/halfedge.hpp"

namespace alluvion {
namespace dg {

template <typename TF>
class TriangleMesh;

template <typename TF>
class FaceContainer;
template <typename TF>
class FaceIterator : public std::iterator<std::random_access_iterator_tag,
                                          std::array<unsigned int, 3>> {
 public:
  typedef FaceIterator _Mytype;

  FaceIterator() = delete;

  reference operator*() { return m_mesh->face(m_index); }

  bool operator<(_Mytype const& other) const { return m_index < other.m_index; }
  bool operator==(_Mytype const& other) const {
    return m_index == other.m_index;
  }

  bool operator!=(_Mytype const& other) const { return !(*this == other); }

  inline _Mytype& operator++() {
    ++m_index;
    return *this;
  }
  inline _Mytype& operator--() {
    --m_index;
    return *this;
  }

  inline _Mytype operator+(_Mytype const& rhs) {
    return _Mytype(m_index + rhs.m_index, m_mesh);
  }
  inline difference_type operator-(_Mytype const& rhs) {
    return m_index - rhs.m_index;
  }
  inline _Mytype operator-(int const& rhs) {
    return _Mytype(m_index - rhs, m_mesh);
  }

  unsigned int vertex(unsigned int i) const {
    return m_mesh->faceVertex(m_index, i);
  }
  unsigned int& vertex(unsigned int i) {
    return m_mesh->faceVertex(m_index, i);
  }

 private:
  friend class FaceContainer<TF>;
  FaceIterator(unsigned int index, TriangleMesh<TF>* mesh)
      : m_index(index), m_mesh(mesh) {}

  unsigned int m_index;
  TriangleMesh<TF>* m_mesh;
};
template <typename TF>
class FaceConstContainer;
template <typename TF>
class FaceConstIterator
    : public std::iterator<std::random_access_iterator_tag,
                           std::array<unsigned int, 3> const> {
 public:
  typedef FaceConstIterator _Mytype;

  FaceConstIterator() = delete;

  reference operator*() { return m_mesh->face(m_index); }

  bool operator<(_Mytype const& other) const { return m_index < other.m_index; }
  bool operator==(_Mytype const& other) const {
    return m_index == other.m_index;
  }

  bool operator!=(_Mytype const& other) const { return !(*this == other); }

  inline _Mytype& operator++() {
    ++m_index;
    return *this;
  }
  inline _Mytype& operator--() {
    --m_index;
    return *this;
  }

  inline _Mytype operator+(_Mytype const& rhs) const {
    return _Mytype(m_index + rhs.m_index, m_mesh);
  }
  inline difference_type operator-(_Mytype const& rhs) const {
    return m_index - rhs.m_index;
  }
  inline _Mytype operator-(int const& rhs) const {
    return _Mytype(m_index - rhs, m_mesh);
  }

  unsigned int vertex(unsigned int i) const;
  unsigned int& vertex(unsigned int i);

 private:
  friend class FaceConstContainer<TF>;
  FaceConstIterator(unsigned int index, TriangleMesh<TF> const* mesh)
      : m_index(index), m_mesh(mesh) {}

  unsigned int m_index;
  TriangleMesh<TF> const* m_mesh;
};

template <typename TF>
class IncidentFaceContainer;
template <typename TF>
class IncidentFaceIterator
    : public std::iterator<std::forward_iterator_tag, Halfedge<unsigned int>> {
 public:
  typedef IncidentFaceIterator _Mytype;

  value_type operator*() { return m_h; }
  _Mytype& operator++() {
    Halfedge<unsigned int> o = m_mesh->opposite(m_h);
    if (o.isBoundary()) {
      m_h = Halfedge<unsigned int>();
      return *this;
    }
    m_h = o.next();
    if (m_h == m_begin) {
      m_h = Halfedge<unsigned int>();
    }
    return *this;
  }
  bool operator==(_Mytype const& other) const { return m_h == other.m_h; }

  bool operator!=(_Mytype const& other) const { return !(*this == other); }

 private:
  friend class IncidentFaceContainer<TF>;
  IncidentFaceIterator(unsigned int v, TriangleMesh<TF> const* mesh)
      : m_mesh(mesh),
        m_h(mesh->incident_halfedge(v)),
        m_begin(mesh->incident_halfedge(v)) {
    if (m_h.isBoundary()) m_h = mesh->opposite(m_h).next();
  }

  IncidentFaceIterator() : m_h(), m_begin(), m_mesh(nullptr) {}

  Halfedge<unsigned int> m_h, m_begin;
  TriangleMesh<TF> const* m_mesh;
};

template <typename TF>
class VertexContainer;

template <typename TF>
class VertexIterator
    : public std::iterator<std::random_access_iterator_tag, Vector3r<TF>> {
 public:
  // std::iterator<>::reference with templates:
  // https://stackoverflow.com/a/67693261
  using base = typename VertexIterator::iterator;
  typedef VertexIterator _Mytype;

  VertexIterator() = delete;

  typename base::reference operator*() { return m_mesh->vertex(m_index); }

  bool operator<(_Mytype const& other) const { return m_index < other.m_index; }
  bool operator==(_Mytype const& other) const {
    return m_index == other.m_index;
  }

  bool operator!=(_Mytype const& other) const { return !(*this == other); }

  inline _Mytype& operator++() {
    ++m_index;
    return *this;
  }
  inline _Mytype& operator--() {
    --m_index;
    return *this;
  }

  inline _Mytype operator+(_Mytype const& rhs) const {
    return _Mytype(m_index + rhs.m_index, m_mesh);
  }
  inline typename base::difference_type operator-(_Mytype const& rhs) const {
    return m_index - rhs.m_index;
  }
  inline _Mytype operator-(int const& rhs) const {
    return _Mytype(m_index - rhs, m_mesh);
  }

  unsigned int index() const { return m_index; }

 private:
  friend class VertexContainer<TF>;
  VertexIterator(unsigned int index, TriangleMesh<TF>* mesh)
      : m_index(index), m_mesh(mesh) {}

  unsigned int m_index;
  TriangleMesh<TF>* m_mesh;
};

template <typename TF>
class VertexConstContainer;
template <typename TF>
class VertexConstIterator
    : public std::iterator<std::random_access_iterator_tag,
                           Vector3r<TF> const> {
 public:
  using base = typename VertexConstIterator::iterator;
  typedef VertexConstIterator _Mytype;

  VertexConstIterator() = delete;

  typename base::reference operator*() { return m_mesh->vertex(m_index); }

  bool operator<(_Mytype const& other) const { return m_index < other.m_index; }
  bool operator==(_Mytype const& other) const {
    return m_index == other.m_index;
  }

  bool operator!=(_Mytype const& other) const { return !(*this == other); }

  inline _Mytype& operator++() {
    ++m_index;
    return *this;
  }
  inline _Mytype& operator--() {
    --m_index;
    return *this;
  }

  inline _Mytype operator+(_Mytype const& rhs) const {
    return _Mytype(m_index + rhs.m_index, m_mesh);
  }
  inline typename base::difference_type operator-(_Mytype const& rhs) const {
    return m_index - rhs.m_index;
  }
  inline _Mytype operator-(int const& rhs) const {
    return _Mytype(m_index - rhs, m_mesh);
  }

  unsigned int index() const;

 private:
  friend class VertexConstContainer<TF>;
  VertexConstIterator(unsigned int index, TriangleMesh<TF> const* mesh)
      : m_index(index), m_mesh(mesh) {}

  unsigned int m_index;
  TriangleMesh<TF> const* m_mesh;
};
}  // namespace dg
}  // namespace alluvion

#endif
