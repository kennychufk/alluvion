#ifndef ALLUVION_STORE_HPP
#define ALLUVION_STORE_HPP

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "alluvion/display.hpp"
#include "alluvion/graphical_variable.hpp"
#include "alluvion/unique_device_pointer.hpp"
#include "alluvion/unique_graphical_resource.hpp"
#include "alluvion/variable.hpp"
namespace alluvion {
class Store {
 private:
  std::unordered_map<void*, UniqueDevicePointer> pointer_dict;
  // NOTE: GraphicalVariable should be destructed before Display
  std::unique_ptr<Display> display_;
  std::unordered_map<GLuint, UniqueGraphicalResource> graphical_resource_dict_;
  std::vector<cudaGraphicsResource*> resource_array_;

  void update_resource_array();

 public:
  Store();
  virtual ~Store();

  Display* create_display(int width, int height, const char* title);

  template <unsigned int D, typename M>
  Variable<D, M> create(std::array<unsigned int, D> const& shape) {
    Variable<D, M> var(shape);
    pointer_dict.emplace(std::piecewise_construct,
                         std::forward_as_tuple(var.ptr_),
                         std::forward_as_tuple(var.ptr_));
    return var;
  }

  template <unsigned int D, typename M>
  GraphicalVariable<D, M> create_graphical(
      std::array<unsigned int, D> const& shape) {
    if (!display_) {
      std::cerr << "Display not created" << std::endl;
      abort();
    }
    GraphicalVariable<D, M> var(shape);
    graphical_resource_dict_.emplace(
        std::piecewise_construct, std::forward_as_tuple(var.vbo_),
        std::forward_as_tuple(var.vbo_, var.res_, &var));
    update_resource_array();
    return var;
  }

  void map_graphical_pointers();
  void unmap_graphical_pointers();
};
}  // namespace alluvion

#endif /* ALLUVION_STORE_HPP */