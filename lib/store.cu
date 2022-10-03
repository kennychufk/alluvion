#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include "alluvion/store.hpp"

namespace alluvion {
Store::Store() : cnf_(Constf{0}), cnd_(Constd{0}), cni_(ConstiN{0}) {}
Store::~Store() {}
void Store::update_resource_array() {
  resource_array_.resize(graphical_resource_dict_.size());
  int i = 0;
  for (auto& entry : graphical_resource_dict_) {
    resource_array_[i++] = entry.second.res_;
  }
}
void Store::set_device(int device) {
  Allocator::abort_if_error(cudaSetDevice(device));
}
Display* Store::create_display(int width, int height, const char* title,
                               bool offscreen) {
  display_.reset(new Display(width, height, title, offscreen));
  return display_.get();
}
Display const* Store::get_display() const { return display_.get(); }
Display* Store::get_display() { return display_.get(); }
bool Store::has_display() const { return static_cast<bool>(display_); }
void Store::map_graphical_pointers() {
  GraphicalAllocator::map(resource_array_);

  for (auto& entry : graphical_resource_dict_) {
    UniqueGraphicalResource& unique_resource = entry.second;
    unique_resource.set_mapped_pointer(
        GraphicalAllocator::get_mapped_pointer(unique_resource.res_));
  }
}
void Store::unmap_graphical_pointers() {
  GraphicalAllocator::unmap(resource_array_);
  for (auto& entry : graphical_resource_dict_) {
    UniqueGraphicalResource& unique_resource = entry.second;
    unique_resource.set_mapped_pointer(nullptr);
  }
}

std::tuple<std::vector<U>, U, char> Store::get_alu_info(const char* filename) {
  std::tuple<std::vector<U>, U, char> result;
  std::vector<U>& shape = std::get<0>(result);
  U& num_primitives_per_element = std::get<1>(result);
  char& type_label = std::get<2>(result);
  std::ifstream stream(filename, std::ios::binary);
  if (!stream.is_open()) {
    std::stringstream error_sstream;
    error_sstream << "Failed to open alu file: " << filename;
    throw std::runtime_error(error_sstream.str());
  }
  while (true) {
    U shape_item;
    stream.read(reinterpret_cast<char*>(&shape_item), sizeof(U));
    if (shape_item == 0) break;
    shape.push_back(shape_item);
  }
  stream.read(reinterpret_cast<char*>(&num_primitives_per_element), sizeof(U));
  stream.read(reinterpret_cast<char*>(&type_label), sizeof(char));
  return result;
}
}  // namespace alluvion
