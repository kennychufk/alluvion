#ifndef ALLUVION_STORE_HPP
#define ALLUVION_STORE_HPP

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "alluvion/constants.hpp"
#include "alluvion/display.hpp"
#include "alluvion/graphical_variable.hpp"
#include "alluvion/pinned_variable.hpp"
#include "alluvion/unique_device_pointer.hpp"
#include "alluvion/unique_graphical_resource.hpp"
#include "alluvion/unique_pinned_pointer.hpp"
#include "alluvion/variable.hpp"
namespace alluvion {
class Store {
 private:
  // NOTE: cannot use std::vector or std::map for unique resources due to
  // reallocation
  std::unordered_map<void*, UniqueDevicePointer> pointer_dict_;
  std::unordered_map<void*, UniquePinnedPointer> pinned_pointer_dict_;
  // NOTE: GraphicalVariable should be destructed before Display
  std::unique_ptr<Display> display_;
  std::unordered_map<GLuint, UniqueGraphicalResource> graphical_resource_dict_;
  std::vector<cudaGraphicsResource*> resource_array_;
  Constf cnf_;
  Constd cnd_;
  ConstiN cni_;

  void update_resource_array();

 public:
  Store();
  virtual ~Store();

  static void set_device(int device);
  Display* create_display(int width, int height, const char* title,
                          bool offscreen = false);
  Display const* get_display() const;
  Display* get_display();
  bool has_display() const;

  template <unsigned int D, typename M>
  Variable<D, M>* create(std::array<unsigned int, D> const& shape) {
    Variable<D, M>* var = new Variable<D, M>(shape);
    if (var->ptr_) {
      pointer_dict_.emplace(std::piecewise_construct,
                            std::forward_as_tuple(var->ptr_),
                            std::forward_as_tuple(var->ptr_));
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
  void remove_graphical(GraphicalVariable<D, M>& var) {
    if (var.vbo_ != 0) {
      graphical_resource_dict_.erase(var.vbo_);
    }
    var.ptr_ = nullptr;
    var.vbo_ = 0;
    var.res_ = nullptr;
    update_resource_array();
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

  /*
   * NOTE: Should return a pointer to the dynamically allocated instance.
   * =================================
   * (A) Reason for returning pointer: The device pointer will be later changed
   * from nullptr to the mapped pointer, so it makes sense to return a pointer
   * instead of a copy.
   * =================================
   * (B) Reason for dynamic allocation: The ownership of the dynamically
   * allocated instance must be transferred to the client to avoid the following
   * 2 unexpected behaviors (reasons unknown):
   * =================================
   * Problem 1: Program crashes upon the invocation of any virtual function
   * (even when invoked internally in C++).
   * =================================
   * Problem 2: Partially-corrupted data are returned when copying data from the
   * device memory in interactive Python (IPython or interactive Python shell).
   */
  template <unsigned int D, typename M>
  GraphicalVariable<D, M>* create_graphical(
      std::array<unsigned int, D> const& shape) {
    if (!display_) {
      std::cerr << "Display not created" << std::endl;
      abort();
    }
    GraphicalVariable<D, M>* var = new GraphicalVariable<D, M>(shape);
    graphical_resource_dict_.emplace(
        std::piecewise_construct, std::forward_as_tuple(var->vbo_),
        std::forward_as_tuple(var->vbo_, var->res_, &(var->ptr_)));
    update_resource_array();
    return var;
  }

  template <typename TF>
  Const<TF>& get_cn() {
    if constexpr (std::is_same_v<TF, float>)
      return cnf_;
    else
      return cnd_;
  }

  ConstiN& get_cni() { return cni_; }

  template <typename TF>
  std::pair<Const<TF>, ConstiN> create_cn() {
    return std::make_pair(Const<TF>(), ConstiN());
  }

  template <typename TF>
  void copy_cn_external(Const<TF> const& cn_arg, ConstiN const& cni_arg) {
    Allocator::abort_if_error(
        cudaMemcpyToSymbol(cni, &cni_arg, sizeof(cni_), 0, cudaMemcpyDefault));
    if constexpr (std::is_same_v<TF, float>)
      Allocator::abort_if_error(
          cudaMemcpyToSymbol(cnf, &cn_arg, sizeof(cnf_), 0, cudaMemcpyDefault));
    else
      Allocator::abort_if_error(
          cudaMemcpyToSymbol(cnd, &cn_arg, sizeof(cnd_), 0, cudaMemcpyDefault));
  }

  template <typename TF>
  void copy_cn() {
    Allocator::abort_if_error(
        cudaMemcpyToSymbol(cni, &cni_, sizeof(cni_), 0, cudaMemcpyDefault));
    if constexpr (std::is_same_v<TF, float>)
      Allocator::abort_if_error(
          cudaMemcpyToSymbol(cnf, &cnf_, sizeof(cnf_), 0, cudaMemcpyDefault));
    else
      Allocator::abort_if_error(
          cudaMemcpyToSymbol(cnd, &cnd_, sizeof(cnd_), 0, cudaMemcpyDefault));
  }

  void map_graphical_pointers();
  void unmap_graphical_pointers();
  static std::tuple<std::vector<U>, U, char> get_alu_info(const char* filename);
};
}  // namespace alluvion

#endif /* ALLUVION_STORE_HPP */
