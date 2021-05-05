#include "alluvion/graphical_allocator.hpp"
#include "alluvion/unique_mesh_buffer.hpp"

namespace alluvion {
UniqueMeshBuffer::UniqueMeshBuffer(MeshBuffer buffer) : mesh_buffer_(buffer) {}

UniqueMeshBuffer::~UniqueMeshBuffer() {
  GraphicalAllocator::free_buffer(&mesh_buffer_.vertex);
  GraphicalAllocator::free_buffer(&mesh_buffer_.index);
}
}  // namespace alluvion
