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
Display const* Store::get_display() const { return display_.get(); }
Display* Store::get_display() { return display_.get(); }
bool Store::has_display() const { return static_cast<bool>(display_); }
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
