#include <tuple>
#include <utility>
#include <vector>

#include "alluvion/store.hpp"

namespace alluvion {
Store::Store() {}
Store::~Store() {}
void Store::update_resource_array() {
  resource_array_.resize(graphical_resource_dict_.size());
  int i = 0;
  for (auto& entry : graphical_resource_dict_) {
    resource_array_[i++] = entry.second.res_;
  }
}
Display* Store::create_display(int width, int height, const char* title) {
  display_.reset(new Display(width, height, title));
  return display_.get();
}
Display* Store::get_display() const { return display_.get(); }
bool Store::has_display() const { return static_cast<bool>(display_); }
MeshBuffer Store::create_mesh_buffer(Mesh const& mesh) {
  if (!display_) {
    std::cerr << "Display not created" << std::endl;
    abort();
  }
  U num_indices = mesh.faces.size() * 3;
  MeshBuffer mesh_buffer(
      GraphicalAllocator::allocate_static_array_buffer<float3>(
          mesh.vertices.size(), mesh.vertices.data()),
      GraphicalAllocator::allocate_static_array_buffer<float3>(
          mesh.normals.size(), mesh.normals.data()),
      GraphicalAllocator::allocate_static_array_buffer<float2>(
          mesh.texcoords.size(), mesh.texcoords.data()),
      GraphicalAllocator::allocate_element_array_buffer<unsigned int>(
          num_indices, mesh.faces.data()),
      num_indices);
  mesh_dict_.emplace(std::piecewise_construct,
                     std::forward_as_tuple(mesh_buffer.vertex),
                     std::forward_as_tuple(mesh_buffer));
  return mesh_buffer;
}
void Store::map_graphical_pointers() {
  GraphicalAllocator::map(resource_array_);

  for (auto& entry : graphical_resource_dict_) {
    UniqueGraphicalResource& unique_resource = entry.second;
    unique_resource.var_->set_pointer(
        GraphicalAllocator::get_mapped_pointer(unique_resource.res_));
  }
}
void Store::unmap_graphical_pointers() {
  GraphicalAllocator::unmap(resource_array_);
  for (auto& entry : graphical_resource_dict_) {
    UniqueGraphicalResource& unique_resource = entry.second;
    unique_resource.var_->set_pointer(nullptr);
  }
}

}  // namespace alluvion
