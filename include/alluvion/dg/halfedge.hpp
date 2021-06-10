#ifndef ALLUVION_DG_HALFEDGE_HPP
#define ALLUVION_DG_HALFEDGE_HPP

#include <cassert>

namespace alluvion {
namespace dg {

template <typename TU>
class Halfedge {
 public:
  Halfedge() : m_code(3) {}
  Halfedge(Halfedge const&) = default;
  Halfedge(TU f, unsigned char e) : m_code((f << 2) | e) {
    // assert(e < 3);
  }

  Halfedge next() const { return Halfedge(face(), (edge() + 1) % 3); }

  Halfedge previous() const { return Halfedge(face(), (edge() + 2) % 3); }

  bool operator==(Halfedge const& other) const {
    return m_code == other.m_code;
  }

  TU face() const { return m_code >> 2; }
  unsigned char edge() const { return m_code & 0x3; }
  bool isBoundary() const { return edge() == 3; }

 private:
  Halfedge(TU code) : m_code(code) {}
  TU m_code;
};
}  // namespace dg
}  // namespace alluvion

#endif
