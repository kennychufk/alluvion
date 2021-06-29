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

}  // namespace alluvion
