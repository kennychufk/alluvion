#ifndef ALLUVION_STORE_HPP
#define ALLUVION_STORE_HPP

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "alluvion/display.hpp"
#include "alluvion/graphical_variable.hpp"
#include "alluvion/mesh.hpp"
#include "alluvion/pinned_variable.hpp"
#include "alluvion/unique_device_pointer.hpp"
#include "alluvion/unique_graphical_resource.hpp"
#include "alluvion/unique_mesh_buffer.hpp"
#include "alluvion/unique_pinned_pointer.hpp"
#include "alluvion/variable.hpp"
namespace alluvion {
class Store {
 private:
  std::unordered_map<void*, UniqueDevicePointer> pointer_dict_;
  std::unordered_map<void*, UniquePinnedPointer> pinned_pointer_dict_;
  // NOTE: GraphicalVariable should be destructed before Display
  std::unique_ptr<Display> display_;
  std::unordered_map<GLuint, UniqueGraphicalResource> graphical_resource_dict_;
  std::vector<cudaGraphicsResource*> resource_array_;
  std::unordered_map<GLuint, UniqueMeshBuffer> mesh_dict_;

  void update_resource_array();

 public:
  Store();
  virtual ~Store();

  Display* create_display(int width, int height, const char* title);
  Display* get_display() const;
  bool has_display() const;

  template <unsigned int D, typename M>
  Variable<D, M> create(std::array<unsigned int, D> const& shape) {
    Variable<D, M> var(shape);
    if (var.ptr_) {
      pointer_dict_.emplace(std::piecewise_construct,
                            std::forward_as_tuple(var.ptr_),
                            std::forward_as_tuple(var.ptr_));
    }
    return var;
  }

  template <unsigned int D, typename M>
  PinnedVariable<D, M> create_pinned(std::array<unsigned int, D> const& shape) {
    PinnedVariable<D, M> var(shape);
    if (var.ptr_) {
      pinned_pointer_dict_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(var.ptr_),
                                   std::forward_as_tuple(var.ptr_));
    }
    return var;
  }

  template <unsigned int D, typename M>
  void remove(Variable<D, M>& var) {
    if (var.ptr_) {
      pointer_dict_.erase(var.ptr_);
    }
    var.ptr_ = nullptr;
  }

  template <unsigned int D, typename M>
  void remove(PinnedVariable<D, M>& var) {
    if (var.ptr_) {
      pinned_pointer_dict_.erase(var.ptr_);
    }
    var.ptr_ = nullptr;
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

  MeshBuffer create_mesh_buffer(Mesh const& mesh);

  void map_graphical_pointers();
  void unmap_graphical_pointers();
};
}  // namespace alluvion

#endif /* ALLUVION_STORE_HPP */
