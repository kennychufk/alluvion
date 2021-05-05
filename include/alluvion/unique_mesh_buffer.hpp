#ifndef ALLUVION_UNIQUE_MESH_BUFFER_HPP
#define ALLUVION_UNIQUE_MESH_BUFFER_HPP

#include "alluvion/mesh_buffer.hpp"

namespace alluvion {
class UniqueMeshBuffer {
 private:
  MeshBuffer mesh_buffer_;

 public:
  UniqueMeshBuffer() = delete;
  UniqueMeshBuffer(MeshBuffer buffer);
  UniqueMeshBuffer(const UniqueMeshBuffer&) = delete;
  virtual ~UniqueMeshBuffer();
};
}  // namespace alluvion

#endif /* ALLUVION_UNIQUE_MESH_BUFFER_HPP */
