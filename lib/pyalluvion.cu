#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <string>

#include "alluvion/constants.hpp"
#include "alluvion/dg/box_distance.hpp"
#include "alluvion/dg/box_shell_distance.hpp"
#include "alluvion/dg/capsule_distance.hpp"
#include "alluvion/dg/cylinder_distance.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/infinite_tube_distance.hpp"
#include "alluvion/dg/mesh_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/display.hpp"
#include "alluvion/display_proxy.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/solver_df.hpp"
#include "alluvion/solver_i.hpp"
#include "alluvion/solver_ii.hpp"
#include "alluvion/solver_pellet.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;
using namespace alluvion::dg;
namespace py = pybind11;

namespace alluvion {
namespace dg {
template <typename TF3, typename TF>
class PyDistance : public Distance<TF3, TF> {
 public:
  using base = Distance<TF3, TF>;
  using base::base;
  TF signedDistance(dg::Vector3r<TF> const& x) const override{
      PYBIND11_OVERLOAD_PURE(TF, base, x)};
  TF3 get_aabb_min() const override {
    PYBIND11_OVERLOAD(TF3, base, get_aabb_min, );
  }
  TF3 get_aabb_max() const override {
    PYBIND11_OVERLOAD(TF3, base, get_aabb_max, );
  }
  TF get_max_distance() const override {
    PYBIND11_OVERLOAD(TF, base, get_max_distance, );
  }
};
}  // namespace dg
}  // namespace alluvion

template <typename TVector, typename TPrimitive>
void declare_vector_operator(py::class_<TVector>& vector_class) {
  vector_class
      .def("__add__",
           [](TVector const& v, const TPrimitive s) { return v + s; })
      .def("__radd__",
           [](TVector const& v, const TPrimitive s) { return s + v; })
      .def("__sub__",
           [](TVector const& v, const TPrimitive s) { return v - s; })
      .def("__rsub__",
           [](TVector const& v, const TPrimitive s) { return s - v; })
      .def("__mul__",
           [](TVector const& v, const TPrimitive s) { return v * s; })
      .def("__rmul__",
           [](TVector const& v, const TPrimitive s) { return s * v; })
      .def("__add__",
           [](TVector const& v, TVector const& v_other) { return v + v_other; })
      .def("__sub__",
           [](TVector const& v, TVector const& v_other) { return v - v_other; })
      .def("__mul__", [](TVector const& v, TVector const& v_other) {
        return v * v_other;
      });

  if constexpr (std::is_same_v<TPrimitive, float> ||
                std::is_same_v<TPrimitive, double>) {
    vector_class
        .def("__truediv__",
             [](TVector const& v, const TPrimitive s) { return v / s; })
        .def("__rtruediv__",
             [](TVector const& v, const TPrimitive s) { return s / v; })
        .def("__truediv__", [](TVector const& v, TVector const& v_other) {
          return v / v_other;
        });
  }
  if constexpr (!std::is_same_v<TPrimitive, uint>) {
    vector_class.def("__neg__", [](TVector const& v) { return -v; });
  }
}

template <typename TVector2, typename TPrimitive>
void declare_vector2(py::module& m, const char* name) {
  py::class_<TVector2> vector_class =
      py::class_<TVector2>(m, name, py::buffer_protocol())
          .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            if (info.ndim != 1)
              throw std::runtime_error("Incompatible buffer dimension!");
            if (info.shape[0] != 2)
              throw std::runtime_error(
                  "Incompatible dimension: expected a vector of 2!");
            TVector2 result;
            if (info.format == py::format_descriptor<float>::format()) {
              float const* ptr = static_cast<float const*>(info.ptr);
              result.x = static_cast<TPrimitive>(ptr[0]);
              result.y = static_cast<TPrimitive>(ptr[1]);
            } else if (info.format == py::format_descriptor<double>::format()) {
              double const* ptr = static_cast<double const*>(info.ptr);
              result.x = static_cast<TPrimitive>(ptr[0]);
              result.y = static_cast<TPrimitive>(ptr[1]);
            } else {
              throw std::runtime_error(
                  "Incompatible format: expected a float/double array!");
            }
            return result;
          }))
          .def_buffer([](TVector2& v) -> py::buffer_info {
            return py::buffer_info(
                &v,                 /* Pointer to buffer */
                sizeof(TPrimitive), /* Size of one scalar */
                py::format_descriptor<TPrimitive>::format(), /* Python
                                                           struct-style format
                                                           descriptor */
                1,                   /* Number of dimensions */
                {2},                 /* Buffer dimensions */
                {sizeof(TPrimitive)} /* Strides (in bytes) for each index */
            );
          })
          .def(py::init<TPrimitive, TPrimitive>())
          .def_static("from_scalar",
                      [](const TPrimitive s) {
                        return TVector2{s, s};
                      })
          .def_readwrite("x", &TVector2::x)
          .def_readwrite("y", &TVector2::y)
          .def("__repr__", [](TVector2 const& v) {
            std::stringstream stream;
            stream << "(" << v.x << ", " << v.y << ", "
                   << ")";
            return stream.str();
          });
  declare_vector_operator<TVector2, TPrimitive>(vector_class);
}

template <typename TVector3, typename TPrimitive>
void declare_vector3(py::module& m, const char* name) {
  py::class_<TVector3> vector_class =
      py::class_<TVector3>(m, name, py::buffer_protocol())
          .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            if (info.ndim != 1)
              throw std::runtime_error("Incompatible buffer dimension!");
            if (info.shape[0] != 3)
              throw std::runtime_error(
                  "Incompatible dimension: expected a vector of 3!");
            TVector3 result;
            if (info.format == py::format_descriptor<float>::format()) {
              float const* ptr = static_cast<float const*>(info.ptr);
              result.x = static_cast<TPrimitive>(ptr[0]);
              result.y = static_cast<TPrimitive>(ptr[1]);
              result.z = static_cast<TPrimitive>(ptr[2]);
            } else if (info.format == py::format_descriptor<double>::format()) {
              double const* ptr = static_cast<double const*>(info.ptr);
              result.x = static_cast<TPrimitive>(ptr[0]);
              result.y = static_cast<TPrimitive>(ptr[1]);
              result.z = static_cast<TPrimitive>(ptr[2]);
            } else {
              throw std::runtime_error(
                  "Incompatible format: expected a float/double array!");
            }
            return result;
          }))
          .def_buffer([](TVector3& v) -> py::buffer_info {
            return py::buffer_info(
                &v,                 /* Pointer to buffer */
                sizeof(TPrimitive), /* Size of one scalar */
                py::format_descriptor<TPrimitive>::format(), /* Python
                                                           struct-style format
                                                           descriptor */
                1,                   /* Number of dimensions */
                {3},                 /* Buffer dimensions */
                {sizeof(TPrimitive)} /* Strides (in bytes) for each index */
            );
          })
          .def(py::init<TPrimitive, TPrimitive, TPrimitive>())
          .def_static("from_scalar",
                      [](const TPrimitive s) {
                        return TVector3{s, s, s};
                      })
          .def_readwrite("x", &TVector3::x)
          .def_readwrite("y", &TVector3::y)
          .def_readwrite("z", &TVector3::z)
          .def("__repr__", [](TVector3 const& v) {
            std::stringstream stream;
            stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
            return stream.str();
          });
  declare_vector_operator<TVector3, TPrimitive>(vector_class);
}

template <typename TVector4, typename TPrimitive>
void declare_vector4(py::module& m, const char* name) {
  py::class_<TVector4> vector_class =
      py::class_<TVector4>(m, name, py::buffer_protocol())
          .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            if (info.ndim != 1)
              throw std::runtime_error("Incompatible buffer dimension!");
            if (info.shape[0] != 4)
              throw std::runtime_error(
                  "Incompatible dimension: expected a vector of 4!");
            TVector4 result;
            if (info.format == py::format_descriptor<float>::format()) {
              float const* ptr = static_cast<float const*>(info.ptr);
              result.x = static_cast<TPrimitive>(ptr[0]);
              result.y = static_cast<TPrimitive>(ptr[1]);
              result.z = static_cast<TPrimitive>(ptr[2]);
              result.w = static_cast<TPrimitive>(ptr[3]);
            } else if (info.format == py::format_descriptor<double>::format()) {
              double const* ptr = static_cast<double const*>(info.ptr);
              result.x = static_cast<TPrimitive>(ptr[0]);
              result.y = static_cast<TPrimitive>(ptr[1]);
              result.z = static_cast<TPrimitive>(ptr[2]);
              result.w = static_cast<TPrimitive>(ptr[3]);
            } else {
              throw std::runtime_error(
                  "Incompatible format: expected a float/double array!");
            }
            return result;
          }))
          .def_buffer([](TVector4& v) -> py::buffer_info {
            return py::buffer_info(
                &v,                 /* Pointer to buffer */
                sizeof(TPrimitive), /* Size of one scalar */
                py::format_descriptor<TPrimitive>::format(), /* Python
                                                           struct-style format
                                                           descriptor */
                1,                   /* Number of dimensions */
                {4},                 /* Buffer dimensions */
                {sizeof(TPrimitive)} /* Strides (in bytes) for each index */
            );
          })
          .def(py::init<TPrimitive, TPrimitive, TPrimitive, TPrimitive>())
          .def_static("from_scalar",
                      [](const TPrimitive s) {
                        return TVector4{s, s, s, s};
                      })
          .def_readwrite("x", &TVector4::x)
          .def_readwrite("y", &TVector4::y)
          .def_readwrite("z", &TVector4::z)
          .def_readwrite("w", &TVector4::w)
          .def("__repr__", [](TVector4 const& v) {
            std::stringstream stream;
            stream << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w
                   << ")";
            return stream.str();
          });
  declare_vector_operator<TVector4, TPrimitive>(vector_class);
}

template <unsigned int D, typename M, typename PrimitiveType>
void declare_variable(py::module& m, py::class_<Store>& store_class,
                      py::class_<Runner<float>>* runner_float_class,
                      py::class_<Runner<double>>* runner_double_class,
                      const char* name) {
  using VariableClass = Variable<D, M>;
  using GraphicalVariableClass = GraphicalVariable<D, M>;
  std::string create_func_name = std::string("create") + name;
  std::string remove_func_name = std::string("remove") + name;
  store_class.def(create_func_name.c_str(), &Store::create<D, M>,
                  py::return_value_policy::take_ownership);
  store_class.def(remove_func_name.c_str(),
                  py::overload_cast<Variable<D, M>&>(&Store::remove<D, M>));
  std::string create_graphical_func_name =
      std::string("create_graphical") + name;
  std::string remove_graphical_func_name =
      std::string("remove_graphical") + name;
  store_class.def(create_graphical_func_name.c_str(),
                  &Store::create_graphical<D, M>,
                  py::return_value_policy::take_ownership);
  store_class.def(remove_graphical_func_name.c_str(),
                  &Store::remove_graphical<D, M>);

  if (runner_float_class != nullptr)
    runner_float_class->def_static("sum", &Runner<float>::template sum<D, M>,
                                   py::arg("var"), py::arg("n"),
                                   py::arg("offset") = 0);
  if (runner_double_class != nullptr)
    runner_double_class->def_static("sum", &Runner<double>::template sum<D, M>,
                                    py::arg("var"), py::arg("n"),
                                    py::arg("offset") = 0);

  std::string variable_name = std::string("Variable") + name;
  py::class_<VariableClass> variable_class =
      py::class_<VariableClass>(m, variable_name.c_str())
          .def(py::init<const VariableClass&>())
          .def_readonly("ptr", &VariableClass::ptr_)
          .def(
              "get_bytes",
              [](VariableClass& variable, py::array_t<unsigned char> bytes,
                 U offset) {
                variable.get_bytes(bytes.mutable_data(), bytes.size(), offset);
              },
              py::arg("bytes"), py::arg("offset") = 0)
          .def(
              "set_bytes",
              [](VariableClass& variable, py::array_t<unsigned char> bytes,
                 U offset) {
                variable.set_bytes(bytes.data(), bytes.size(), offset);
              },
              py::arg("bytes"), py::arg("offset") = 0)
          .def("set_from",
               py::overload_cast<VariableClass const&, U, U>(
                   &VariableClass::set_from),
               py::arg("src"), py::arg("num_elements"), py::arg("offset") = 0)
          .def(
              "set_from",
              py::overload_cast<VariableClass const&>(&VariableClass::set_from),
              py::arg("src"))
          .def("set_zero", &VariableClass::set_zero)
          .def("set_same",
               py::overload_cast<int, U, U>(&VariableClass::set_same),
               py::arg("value"), py::arg("num_elements"), py::arg("offset") = 0)
          .def("set_same", py::overload_cast<int>(&VariableClass::set_same),
               py::arg("value"))
          .def("scale", py::overload_cast<M>(&VariableClass::template scale<M>))
          .def("scale",
               py::overload_cast<M, U, U>(&VariableClass::template scale<M>),
               py::arg("multiplier"), py::arg("num_elements"),
               py::arg("offset") = 0)
          .def("shift", py::overload_cast<M>(&VariableClass::shift))
          .def("shift", py::overload_cast<M, U, U>(&VariableClass::shift),
               py::arg("addend"), py::arg("num_elements"),
               py::arg("offset") = 0)
          .def("fill", py::overload_cast<M, U, U>(&VariableClass::fill),
               py::arg("value"), py::arg("num_elements"), py::arg("offset") = 0)
          .def("fill", py::overload_cast<M>(&VariableClass::fill),
               py::arg("value"))
          .def("get_type", &VariableClass::get_type)
          .def("get_num_primitives_per_element",
               &VariableClass::get_num_primitives_per_element)
          .def("get_linear_shape", &VariableClass::get_linear_shape)
          .def("get_num_primitives", &VariableClass::get_num_primitives)
          .def("read_file", &VariableClass::read_file)
          .def("write_file", &VariableClass::write_file, py::arg("filename"),
               py::arg("shape_outermost") = 0, py::arg("offset") = 0)
          .def("get_shape", &VariableClass::get_shape);

  if (typeid(PrimitiveType) != typeid(bool)) {
    variable_class
        .def("scale", py::overload_cast<PrimitiveType>(
                          &VariableClass::template scale<PrimitiveType>))
        .def("scale",
             py::overload_cast<PrimitiveType, U, U>(
                 &VariableClass::template scale<PrimitiveType>),
             py::arg("multiplier"), py::arg("num_elements"),
             py::arg("offset") = 0);
  }
  std::string graphical_variable_name = std::string("GraphicalVariable") + name;
  py::class_<GraphicalVariableClass, VariableClass>(
      m, graphical_variable_name.c_str())
      .def(py::init<const GraphicalVariableClass&>())
      .def_readonly("vbo", &GraphicalVariableClass::vbo_);
}

template <unsigned int D, typename M, typename TPrimitive>
void declare_metrics(py::module& m,
                     py::class_<Runner<TPrimitive>>* runner_class) {
  runner_class->def_static(
      "calculate_se_weighted",
      &Runner<TPrimitive>::template calculate_se_weighted<D, M, TPrimitive>,
      py::arg("v0"), py::arg("v1"), py::arg("weight0"), py::arg("n"),
      py::arg("offset") = 0);
  runner_class->def_static(
      "calculate_se",
      &Runner<TPrimitive>::template calculate_se<D, M, TPrimitive>,
      py::arg("v0"), py::arg("v1"), py::arg("se"), py::arg("n"),
      py::arg("offset") = 0);
  runner_class->def_static(
      "calculate_se_yz_masked",
      &Runner<TPrimitive>::template calculate_se_yz_masked<D, M, TPrimitive>,
      py::arg("v0"), py::arg("v1"), py::arg("mask"), py::arg("n"),
      py::arg("offset") = 0);
  runner_class->def_static(
      "calculate_kl_divergence", &Runner<TPrimitive>::calculate_kl_divergence,
      py::arg("histogram_p"), py::arg("histogram_q"), py::arg("n_p"),
      py::arg("n_q"), py::arg("q_lower_bound") = static_cast<TPrimitive>(1e-6),
      py::arg("num_bins") = kHistogram256BinCount);
  runner_class->def_static(
      "sum_products", &Runner<TPrimitive>::template sum_products<D, TPrimitive>,
      py::arg("v0"), py::arg("v1"), py::arg("n"), py::arg("offset") = 0);
  runner_class->def_static(
      "sum_products_different_offsets",
      &Runner<TPrimitive>::template sum_products_different_offsets<D, 2,
                                                                   TPrimitive>,
      py::arg("v0"), py::arg("v1"), py::arg("n"), py::arg("offset0") = 0,
      py::arg("offset1") = 0);
}

template <unsigned int D, typename M>
void declare_pinned_variable(py::module& m, const char* name) {
  using PinnedVariableClass = PinnedVariable<D, M>;
  std::string class_name = std::string("PinnedVariable") + name;
  py::class_<PinnedVariableClass>(m, class_name.c_str())
      .def(py::init<const PinnedVariableClass&>())
      .def(
          "get_bytes",
          [](PinnedVariableClass& variable, py::array_t<unsigned char> bytes,
             U offset) {
            variable.get_bytes(bytes.mutable_data(), bytes.size(), offset);
          },
          py::arg("bytes"), py::arg("offset") = 0)
      .def(
          "set_bytes",
          [](PinnedVariableClass& variable, py::array_t<unsigned char> bytes,
             U offset) {
            variable.set_bytes(bytes.data(), bytes.size(), offset);
          },
          py::arg("bytes"), py::arg("offset") = 0)
      .def("__getitem__",
           [](PinnedVariableClass& variable, U key) {
             if (key >= variable.get_linear_shape()) {
               std::stringstream error_sstream;
               error_sstream << "PinnedVariable: index out of range (" << key
                             << " >= " << variable.get_linear_shape() << ")"
                             << std::endl;
               throw std::runtime_error(error_sstream.str());
             }
             return variable(key);
           })
      .def("__setitem__",
           [](PinnedVariableClass& variable, U key, M const& v) {
             if (key >= variable.get_linear_shape()) {
               std::stringstream error_sstream;
               error_sstream << "PinnedVariable: index out of range (" << key
                             << " >= " << variable.get_linear_shape() << ")"
                             << std::endl;
               throw std::runtime_error(error_sstream.str());
             }
             variable(key) = v;
           })
      .def("get_type", &PinnedVariableClass::get_type)
      .def("get_num_primitives_per_element",
           &PinnedVariableClass::get_num_primitives_per_element)
      .def("get_linear_shape", &PinnedVariableClass::get_linear_shape)
      .def("get_num_primitives", &PinnedVariableClass::get_num_primitives)
      .def("get_shape", &PinnedVariableClass::get_shape);
}

template <typename TF>
void declare_pile(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  std::string class_name = std::string("Pile") + name;
  py::class_<TPile>(m, class_name.c_str())
      .def(py::init<Store&, TRunner&, U, VolumeMethod, U, Const<TF>*,
                    ConstiN*>(),
           py::arg("store"), py::arg("runner"), py::arg("max_num_contacts"),
           py::arg("volume_method") = VolumeMethod::volume_map,
           py::arg("max_num_pellets") = 10000, py::arg("cn") = nullptr,
           py::arg("cni") = nullptr)
      .def_readwrite("mass", &TPile::mass_)
      .def_readwrite("x", &TPile::x_)
      .def_readwrite("v", &TPile::v_)
      .def_readwrite("q", &TPile::q_)
      .def_readwrite("omega", &TPile::omega_)
      .def_readwrite("restitution", &TPile::restitution_)
      .def_readwrite("friction", &TPile::friction_)
      .def_readwrite("inertia_tensor", &TPile::inertia_tensor_)
      .def_readonly("domain_min_list", &TPile::domain_min_list_)
      .def_readonly("domain_max_list", &TPile::domain_max_list_)
      .def_readonly("resolution_list", &TPile::resolution_list_)
      .def_readonly("cell_size_list", &TPile::cell_size_list_)
      .def_readonly("sign_list", &TPile::sign_list_)
      .def_readonly("grid_size_list", &TPile::grid_size_list_)
      .def_readonly("num_pellets", &TPile::num_pellets_)
      .def_readonly("num_collision_pellets", &TPile::num_collision_pellets_)
      .def_property_readonly("distance_grids", &TPile::get_distance_grids)
      .def_property_readonly("volume_grids", &TPile::get_volume_grids)
      .def_property_readonly(
          "num_contacts_pinned",
          [](TPile const& pile) { return pile.num_contacts_pinned_(0); })
      .def_property_readonly(
          "num_contacts",
          [](TPile const& pile) { return pile.num_contacts_.get(); })
      .def_property_readonly(
          "collision_vertex_list",
          [](TPile const& pile) {
            std::vector<Variable<1, TF3> const*> result;
            result.reserve(pile.collision_vertex_list_.size());
            for (std::unique_ptr<Variable<1, TF3>> const& collision_vertex :
                 pile.collision_vertex_list_) {
              result.push_back(collision_vertex.get());
            }
            return result;
          })
      .def_property_readonly(
          "x_device", [](TPile const& pile) { return pile.x_device_.get(); })
      .def_property_readonly(
          "v_device", [](TPile const& pile) { return pile.v_device_.get(); })
      .def_property_readonly(
          "omega_device",
          [](TPile const& pile) { return pile.omega_device_.get(); })
      .def("add_pellets",
           py::overload_cast<dg::Distance<TF3, TF>*, U3, TF, TF, TF,
                             Variable<1, TF3> const&, TF, TF, TF, TF3 const&,
                             TF3 const&, TQ const&, Mesh const&>(&TPile::add),
           py::arg("distance"), py::arg("resolution") = U3{0},
           py::arg("cell_width") = 0,
           py::arg("margin") = std::numeric_limits<TF>::infinity(),
           py::arg("sign") = 1, py::arg("pellets") = Variable<1, TF3>(),
           py::arg("mass") = 0, py::arg("restitution") = 1,
           py::arg("friction") = 0, py::arg("inertia_tensor") = TF3{1, 1, 1},
           py::arg("x") = TF3{0}, py::arg("q") = TQ{0, 0, 0, 1},
           py::arg("display_mesh") = Mesh())
      .def("add",
           py::overload_cast<dg::Distance<TF3, TF>*, U3, TF, TF, TF,
                             Mesh const&, TF, TF, TF, TF3 const&, TF3 const&,
                             TQ const&, Mesh const&>(&TPile::add),
           py::arg("distance"), py::arg("resolution") = U3{0},
           py::arg("cell_width") = 0,
           py::arg("margin") = std::numeric_limits<TF>::infinity(),
           py::arg("sign") = 1, py::arg("collision_mesh") = Mesh(),
           py::arg("mass") = 0, py::arg("restitution") = 1,
           py::arg("friction") = 0, py::arg("inertia_tensor") = TF3{1, 1, 1},
           py::arg("x") = TF3{0}, py::arg("q") = TQ{0, 0, 0, 1},
           py::arg("display_mesh") = Mesh())
      .def("replace_pellets",
           py::overload_cast<U, dg::Distance<TF3, TF>*, U3, TF, TF, TF,
                             Variable<1, TF3> const&, TF, TF, TF, TF3 const&,
                             TF3 const&, TQ const&, Mesh const&>(
               &TPile::replace),
           py::arg("i"), py::arg("distance"), py::arg("resolution") = U3{0},
           py::arg("cell_width") = 0,
           py::arg("margin") = std::numeric_limits<TF>::infinity(),
           py::arg("sign") = 1, py::arg("pellets") = Variable<1, TF3>(),
           py::arg("mass") = 0, py::arg("restitution") = 1,
           py::arg("friction") = 0, py::arg("inertia_tensor") = TF3{1, 1, 1},
           py::arg("x") = TF3{0, 0, 0}, py::arg("q") = TQ{0, 0, 0, 1},
           py::arg("display_mesh") = Mesh())
      .def("replace",
           py::overload_cast<U, dg::Distance<TF3, TF>*, U3, TF, TF, TF,
                             Mesh const&, TF, TF, TF, TF3 const&, TF3 const&,
                             TQ const&, Mesh const&>(&TPile::replace),
           py::arg("i"), py::arg("distance"), py::arg("resolution") = U3{0},
           py::arg("cell_width") = 0,
           py::arg("margin") = std::numeric_limits<TF>::infinity(),
           py::arg("sign") = 1, py::arg("collision_mesh") = Mesh(),
           py::arg("mass") = 0, py::arg("restitution") = 1,
           py::arg("friction") = 0, py::arg("inertia_tensor") = TF3{1, 1, 1},
           py::arg("x") = TF3{0, 0, 0}, py::arg("q") = TQ{0, 0, 0, 1},
           py::arg("display_mesh") = Mesh())
      .def("compute_sort_custom_beads_internal_all",
           &TPile::compute_sort_custom_beads_internal_all,
           py::arg("internal_encoded"), py::arg("bead_x"))
      .def("compute_sort_fluid_block_internal_all",
           &TPile::compute_sort_fluid_block_internal_all,
           py::arg("internal_encoded"), py::arg("box_min"), py::arg("box_max"),
           py::arg("particle_radius"), py::arg("mode") = 0)
      .def("compute_sort_fluid_cylinder_internal_all",
           &TPile::compute_sort_fluid_cylinder_internal_all,
           py::arg("internal_encoded"), py::arg("radius"),
           py::arg("particle_radius"), py::arg("y_min"), py::arg("y_max"))
      .def("set_gravity", &TPile::set_gravity)
      .def("reallocate_kinematics_on_device",
           &TPile::reallocate_kinematics_on_device)
      .def("reallocate_kinematics_on_pinned",
           &TPile::reallocate_kinematics_on_pinned)
      .def("write_file", &TPile::write_file, py::arg("filename"),
           py::arg("x_scale") = 1, py::arg("v_scale") = 1,
           py::arg("omega_scale") = 1)
      .def("read_file", &TPile::read_file, py::arg("filename"),
           py::arg("num_rigids") = -1, py::arg("dst_offset") = 0,
           py::arg("src_offset") = 0)
      .def_static("get_size_from_file", &TPile::get_size_from_file,
                  py::arg("filename"))
      .def("copy_kinematics_to_device", &TPile::copy_kinematics_to_device)
      .def("hint_identical_sequence", &TPile::hint_identical_sequence,
           py::arg("begin_id"), py::arg("end_id"))
      .def("integrate_kinematics", &TPile::integrate_kinematics)
      .def("calculate_cfl_v2", &TPile::calculate_cfl_v2)
      .def("find_contacts", py::overload_cast<U, U>(&TPile::find_contacts))
      .def("find_contacts", py::overload_cast<U>(&TPile::find_contacts))
      .def("find_contacts",
           py::overload_cast<Variable<1, TF3> const&, U>(&TPile::find_contacts))
      .def("solve_contacts", &TPile::solve_contacts)
      .def("get_size", &TPile::get_size)
      .def("get_matrix", &TPile::get_matrix)
      .def("compute_mask", &TPile::compute_mask);
}

template <typename TF3, typename TF>
void declare_distance(py::module& m, const char* name) {
  using TDistance = dg::Distance<TF3, TF>;
  std::string class_name = std::string("Distance") + name;
  py::class_<TDistance, dg::PyDistance<TF3, TF>>(m, class_name.c_str())
      .def_readonly("aabb_min", &TDistance::aabb_min_)
      .def_readonly("aabb_max", &TDistance::aabb_max_)
      .def_readonly("max_distance", &TDistance::max_distance_);
}

template <typename TF3, typename TF>
void declare_sphere_distance(py::module& m, const char* name) {
  using TSphereDistance = dg::SphereDistance<TF3, TF>;
  std::string class_name = std::string("SphereDistance") + name;
  py::class_<TSphereDistance, dg::Distance<TF3, TF>>(m, class_name.c_str())
      .def(py::init<TF>())
      .def_static(
          "create", [](TF radius) { return new TSphereDistance(radius); },
          py::return_value_policy::reference);
}

template <typename TF3, typename TF>
void declare_box_distance(py::module& m, const char* name) {
  using TBoxDistance = dg::BoxDistance<TF3, TF>;
  std::string class_name = std::string("BoxDistance") + name;
  py::class_<TBoxDistance, dg::Distance<TF3, TF>>(m, class_name.c_str())
      .def(py::init<TF3, TF>(), py::arg("widths"), py::arg("outset") = 0)
      .def_readonly("half_widths", &TBoxDistance::half_widths)
      .def_readonly("outset", &TBoxDistance::outset)
      .def_static(
          "create",
          [](TF3 widths, TF outset) {
            return new TBoxDistance(widths, outset);
          },
          py::return_value_policy::reference, py::arg("widths"),
          py::arg("outset") = 0);
}

template <typename TF3, typename TF>
void declare_box_shell_distance(py::module& m, const char* name) {
  using TBoxShellDistance = dg::BoxShellDistance<TF3, TF>;
  std::string class_name = std::string("BoxShellDistance") + name;
  py::class_<TBoxShellDistance, dg::Distance<TF3, TF>>(m, class_name.c_str())
      .def(py::init<TF3, TF, TF>(), py::arg("widths"), py::arg("thickness"),
           py::arg("outset") = 0)
      .def_readonly("half_inner_widths", &TBoxShellDistance::half_inner_widths)
      .def_readonly("half_outer_widths", &TBoxShellDistance::half_outer_widths)
      .def_readonly("outset", &TBoxShellDistance::outset)
      .def_static(
          "create",
          [](TF3 widths, TF thickness, TF outset) {
            return new TBoxShellDistance(widths, thickness, outset);
          },
          py::return_value_policy::reference, py::arg("widths"),
          py::arg("thickness"), py::arg("outset") = 0);
}

template <typename TF3, typename TF>
void declare_cylinder_distance(py::module& m, const char* name) {
  using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
  std::string class_name = std::string("CylinderDistance") + name;
  py::class_<TCylinderDistance, dg::Distance<TF3, TF>>(m, class_name.c_str())
      .def(py::init<TF, TF, TF>(), py::arg("radius"), py::arg("height"),
           py::arg("com_y") = 0)
      .def_static(
          "create",
          [](TF radius, TF height, TF com_y) {
            return new TCylinderDistance(radius, height, com_y);
          },
          py::return_value_policy::reference, py::arg("radius"),
          py::arg("height"), py::arg("com_y") = 0);
}

template <typename TF3, typename TF>
void declare_infinite_cylinder_distance(py::module& m, const char* name) {
  using TInfiniteCylinderDistance = dg::InfiniteCylinderDistance<TF3, TF>;
  std::string class_name = std::string("InfiniteCylinderDistance") + name;
  py::class_<TInfiniteCylinderDistance, dg::Distance<TF3, TF>>(
      m, class_name.c_str())
      .def(py::init<TF>())
      .def_static(
          "create",
          [](TF radius) { return new TInfiniteCylinderDistance(radius); },
          py::return_value_policy::reference);
}

template <typename TF3, typename TF>
void declare_infinite_tube_distance(py::module& m, const char* name) {
  using TInfiniteTubeDistance = dg::InfiniteTubeDistance<TF3, TF>;
  std::string class_name = std::string("InfiniteTubeDistance") + name;
  py::class_<TInfiniteTubeDistance, dg::Distance<TF3, TF>>(m,
                                                           class_name.c_str())
      .def(py::init<TF, TF, TF>(), py::arg("inner_radius"),
           py::arg("outer_radius"), py::arg("aabb_half_length") = 0)
      .def_static(
          "create",
          [](TF inner_radius, TF outer_radius, TF aabb_half_length) {
            return new TInfiniteTubeDistance(inner_radius, outer_radius,
                                             aabb_half_length);
          },
          py::arg("inner_radius"), py::arg("outer_radius"),
          py::arg("aabb_half_length") = 0, py::return_value_policy::reference);
}

template <typename TF3, typename TF>
void declare_capsule_distance(py::module& m, const char* name) {
  using TCapsuleDistance = dg::CapsuleDistance<TF3, TF>;
  std::string class_name = std::string("CapsuleDistance") + name;
  py::class_<TCapsuleDistance, dg::Distance<TF3, TF>>(m, class_name.c_str())
      .def(py::init<TF, TF>())
      .def_static(
          "create",
          [](TF radius, TF height) {
            return new TCapsuleDistance(radius, height);
          },
          py::return_value_policy::reference);
}

template <typename TF3, typename TF>
void declare_mesh_distance(py::module& m, const char* name) {
  using TMeshDistance = dg::MeshDistance<TF3, TF>;
  std::string class_name = std::string("MeshDistance") + name;
  py::class_<TMeshDistance, dg::Distance<TF3, TF>>(m, class_name.c_str())
      .def(py::init<TriangleMesh<TF> const&, TF, bool>(), py::arg("mesh"),
           py::arg("offset"), py::arg("precompute_normals") = true)
      .def_static(
          "create",
          [](TriangleMesh<TF> const& mesh, TF offset, bool precompute_normals) {
            return new TMeshDistance(mesh, offset, precompute_normals);
          },
          py::return_value_policy::reference, py::arg("mesh"),
          py::arg("offset"), py::arg("precompute_normals") = true);
}

template <typename TF>
void declare_triangle_mesh(py::module& m, const char* name) {
  using TTriangleMesh = dg::TriangleMesh<TF>;
  std::string class_name = std::string("TriangleMesh") + name;
  py::class_<TTriangleMesh>(m, class_name.c_str())
      .def(py::init<>())
      .def(py::init<std::vector<dg::Vector3r<TF>> const&,
                    std::vector<std::array<unsigned int, 3>> const&>(),
           py::arg("vertices"), py::arg("faces"))
      .def(py::init<std::string const&>(), py::arg("path"));
}

template <typename TF>
void declare_const(py::module& m, const char* name) {
  using TConst = Const<TF>;
  std::string class_name = std::string("Const") + name;
  py::class_<TConst>(m, class_name.c_str())
      .def("set_kernel_radius", &TConst::set_kernel_radius)
      .def("set_particle_attr", &TConst::set_particle_attr)
      .def("set_wrap_length", &TConst::set_wrap_length)
      .def_readonly("kernel_radius", &TConst::kernel_radius)
      .def_readonly("particle_radius", &TConst::particle_radius)
      .def_readonly("particle_vol", &TConst::particle_vol)
      .def_readonly("particle_mass", &TConst::particle_mass)
      .def_readonly("density0", &TConst::density0)
      .def_readonly("wrap_length", &TConst::wrap_length)
      .def_readonly("wrap_min", &TConst::wrap_min)
      .def_readonly("wrap_max", &TConst::wrap_max)
      .def_readwrite("viscosity", &TConst::viscosity)
      .def_readwrite("boundary_viscosity", &TConst::boundary_viscosity)
      .def_readwrite("vorticity_coeff", &TConst::vorticity_coeff)
      .def_readwrite("boundary_vorticity_coeff",
                     &TConst::boundary_vorticity_coeff)
      .def_readwrite("inertia_inverse", &TConst::inertia_inverse)
      .def_readwrite("viscosity_omega", &TConst::viscosity_omega)
      .def_readwrite("surface_tension_coeff", &TConst::surface_tension_coeff)
      .def_readwrite("surface_tension_boundary_coeff",
                     &TConst::surface_tension_boundary_coeff)
      .def_readwrite("gravity", &TConst::gravity)
      .def_readwrite("boundary_epsilon", &TConst::boundary_epsilon)
      .def_readwrite("dfsph_factor_epsilon", &TConst::dfsph_factor_epsilon)
      .def_readwrite("contact_tolerance", &TConst::contact_tolerance);
}

template <typename TF>
void declare_solver(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  using TSolver = Solver<TF>;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  std::string class_name = std::string("Solver") + name;
  py::class_<TSolver>(m, class_name.c_str())
      .def(py::init<TRunner&, TPile&, Store&, U, U, bool, bool,
                    Const<TF> const*, ConstiN const*, bool>(),
           py::arg("runner"), py::arg("pile"), py::arg("store"),
           py::arg("max_num_particles"), py::arg("num_ushers") = 0,
           py::arg("enable_surface_tension") = false,
           py::arg("enable_vorticity") = false, py::arg("cn") = nullptr,
           py::arg("cni") = nullptr, py::arg("graphical") = false)
      .def_readonly("max_num_particles", &TSolver::max_num_particles)
      .def_readonly("particle_max_v2", &TSolver::particle_max_v2)
      .def_readonly("pile_max_v2", &TSolver::pile_max_v2)
      .def_readonly("max_v2", &TSolver::max_v2)
      .def_readonly("cfl_dt", &TSolver::cfl_dt)
      .def_readonly("utilized_cfl", &TSolver::utilized_cfl)
      .def_readwrite("num_particles", &TSolver::num_particles)
      .def_readwrite("t", &TSolver::t)
      .def_readwrite("dt", &TSolver::dt)
      .def_readwrite("initial_dt", &TSolver::initial_dt)
      .def_readwrite("max_dt", &TSolver::max_dt)
      .def_readwrite("min_dt", &TSolver::min_dt)
      .def_readwrite("cfl", &TSolver::cfl)
      .def_readwrite("particle_radius", &TSolver::particle_radius)
      .def_readwrite("enable_surface_tension", &TSolver::enable_surface_tension)
      .def_readwrite("enable_vorticity", &TSolver::enable_vorticity)
      .def_readwrite("next_emission_t", &TSolver::next_emission_t)
      .def_property_readonly(
          "runner", [](TSolver const& solver) { return solver.runner; },
          py::return_value_policy::reference)
      .def_property_readonly(
          "usher", [](TSolver const& solver) { return solver.usher.get(); })
      .def_property_readonly(
          "particle_x",
          [](TSolver const& solver) { return solver.particle_x.get(); })
      .def_property_readonly(
          "particle_v",
          [](TSolver const& solver) { return solver.particle_v.get(); })
      .def_property_readonly(
          "particle_a",
          [](TSolver const& solver) { return solver.particle_a.get(); })
      .def_property_readonly(
          "particle_density",
          [](TSolver const& solver) { return solver.particle_density.get(); })
      .def_property_readonly(
          "particle_boundary",
          [](TSolver const& solver) { return solver.particle_boundary.get(); })
      .def_property_readonly("particle_boundary_kernel",
                             [](TSolver const& solver) {
                               return solver.particle_boundary_kernel.get();
                             })
      .def_property_readonly(
          "particle_force",
          [](TSolver const& solver) { return solver.particle_force.get(); })
      .def_property_readonly(
          "particle_torque",
          [](TSolver const& solver) { return solver.particle_torque.get(); })
      .def_property_readonly(
          "particle_cfl_v2",
          [](TSolver const& solver) { return solver.particle_cfl_v2.get(); })
      .def_property_readonly(
          "particle_normal",
          [](TSolver const& solver) { return solver.particle_normal.get(); })
      .def_property_readonly(
          "particle_omega",
          [](TSolver const& solver) { return solver.particle_omega.get(); })
      .def_property_readonly(
          "particle_angular_acceleration",
          [](TSolver const& solver) {
            return solver.particle_angular_acceleration.get();
          })
      .def_property_readonly(
          "pid", [](TSolver const& solver) { return solver.pid.get(); })
      .def_property_readonly(
          "pid_length",
          [](TSolver const& solver) { return solver.pid_length.get(); })
      .def_property_readonly(
          "particle_neighbors",
          [](TSolver const& solver) { return solver.particle_neighbors.get(); })
      .def_property_readonly("particle_num_neighbors",
                             [](TSolver const& solver) {
                               return solver.particle_num_neighbors.get();
                             })
      .def_property_readonly("particle_boundary_neighbors",
                             [](TSolver const& solver) {
                               return solver.particle_boundary_neighbors.get();
                             })
      .def_property_readonly(
          "particle_num_boundary_neighbors",
          [](TSolver const& solver) {
            return solver.particle_num_boundary_neighbors.get();
          })
      .def_property_readonly(
          "grid_anomaly",
          [](TSolver const& solver) { return solver.grid_anomaly.get(); })
      .def_property_readonly("pile",
                             [](TSolver const& solver) { return &solver.pile; })
      .def("normalize",
           py::overload_cast<Variable<1, TF3> const*, Variable<1, TF>*, TF, TF>(
               &TSolver::normalize))
      .def("normalize",
           py::overload_cast<Variable<1, TF> const*, Variable<1, TF>*, TF, TF>(
               &TSolver::normalize))
      .def("reset_solving_var", &TSolver::reset_solving_var)
      .def("reset_t", &TSolver::reset_t)
      .def("compute_all_boundaries", &TSolver::compute_all_boundaries)
      .def("transform_all_pellets", &TSolver::transform_all_pellets)
      .def("sample_all_boundaries", &TSolver::sample_all_boundaries)
      .def("update_particle_neighbors",
           &TSolver::template update_particle_neighbors<0>)
      .def("update_particle_neighbors_wrap1",
           &TSolver::template update_particle_neighbors<1>);
}

template <typename TF>
void declare_solver_df(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TSolver = Solver<TF>;
  using TSolverDf = SolverDf<TF>;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  std::string class_name = std::string("SolverDf") + name;
  py::class_<TSolverDf, TSolver>(m, class_name.c_str())
      .def(py::init<TRunner&, TPile&, Store&, U, U, bool, bool,
                    Const<TF> const*, ConstiN const*, bool>(),
           py::arg("runner"), py::arg("pile"), py::arg("store"),
           py::arg("max_num_particles"), py::arg("num_ushers") = 0,
           py::arg("enable_surface_tension") = false,
           py::arg("enable_vorticity") = false, py::arg("cn") = nullptr,
           py::arg("cni") = nullptr, py::arg("graphical") = false)
      .def_readonly("num_divergence_solve", &TSolverDf::num_divergence_solve)
      .def_readonly("num_density_solve", &TSolverDf::num_density_solve)
      .def_readonly("mean_density_change", &TSolverDf::mean_density_change)
      .def_readonly("mean_density_error", &TSolverDf::mean_density_error)
      .def_readwrite("enable_divergence_solve",
                     &TSolverDf::enable_divergence_solve)
      .def_readwrite("enable_density_solve", &TSolverDf::enable_density_solve)
      .def_readwrite("density_change_tolerance",
                     &TSolverDf::density_change_tolerance)
      .def_readwrite("density_error_tolerance",
                     &TSolverDf::density_error_tolerance)
      .def_readwrite("min_density_solve", &TSolverDf::min_density_solve)
      .def_readwrite("max_density_solve", &TSolverDf::max_density_solve)
      .def_readwrite("min_divergence_solve", &TSolverDf::min_divergence_solve)
      .def_readwrite("max_divergence_solve", &TSolverDf::max_divergence_solve)
      .def_property_readonly("particle_dfsph_factor",
                             [](TSolverDf const& solver) {
                               return solver.particle_dfsph_factor.get();
                             })
      .def_property_readonly(
          "particle_kappa",
          [](TSolverDf const& solver) { return solver.particle_kappa.get(); })
      .def_property_readonly(
          "particle_kappa_v",
          [](TSolverDf const& solver) { return solver.particle_kappa_v.get(); })
      .def_property_readonly("particle_density_adv",
                             [](TSolverDf const& solver) {
                               return solver.particle_density_adv.get();
                             })
      .def("step", &TSolverDf::template step<0>)
      .def("step_wrap1", &TSolverDf::template step<1>);
}

template <typename TF>
void declare_solver_ii(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TSolver = Solver<TF>;
  using TSolverIi = SolverIi<TF>;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  std::string class_name = std::string("SolverIi") + name;
  py::class_<TSolverIi, TSolver>(m, class_name.c_str())
      .def(py::init<TRunner&, TPile&, Store&, U, U, bool, bool,
                    Const<TF> const*, ConstiN const*, bool>(),
           py::arg("runner"), py::arg("pile"), py::arg("store"),
           py::arg("max_num_particles"), py::arg("num_ushers") = 0,
           py::arg("enable_surface_tension") = false,
           py::arg("enable_vorticity") = false, py::arg("cn") = nullptr,
           py::arg("cni") = nullptr, py::arg("graphical") = false)
      .def_readonly("num_density_solve", &TSolverIi::num_density_solve)
      .def_readonly("mean_density_error", &TSolverIi::mean_density_error)
      .def_readwrite("density_error_tolerance",
                     &TSolverIi::density_error_tolerance)
      .def_readwrite("min_density_solve", &TSolverIi::min_density_solve)
      .def_readwrite("max_density_solve", &TSolverIi::max_density_solve)
      .def_property_readonly("particle_pressure",
                             [](TSolverIi const& solver) {
                               return solver.particle_pressure.get();
                             })
      .def_property_readonly("particle_last_pressure",
                             [](TSolverIi const& solver) {
                               return solver.particle_last_pressure.get();
                             })
      .def_property_readonly(
          "particle_aii",
          [](TSolverIi const& solver) { return solver.particle_aii.get(); })
      .def_property_readonly(
          "particle_dii",
          [](TSolverIi const& solver) { return solver.particle_dii.get(); })
      .def_property_readonly(
          "particle_dij_pj",
          [](TSolverIi const& solver) { return solver.particle_dij_pj.get(); })
      .def_property_readonly("particle_adv_density",
                             [](TSolverIi const& solver) {
                               return solver.particle_adv_density.get();
                             })
      .def_property_readonly("particle_pressure_accel",
                             [](TSolverIi const& solver) {
                               return solver.particle_pressure_accel.get();
                             })
      .def_property_readonly("particle_density_err",
                             [](TSolverIi const& solver) {
                               return solver.particle_density_err.get();
                             })
      .def("step", &TSolverIi::template step<0>)
      .def("step_wrap1", &TSolverIi::template step<1>);
}

template <typename TF>
void declare_solver_i(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TSolver = Solver<TF>;
  using TSolverI = SolverI<TF>;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  std::string class_name = std::string("SolverI") + name;
  py::class_<TSolverI, TSolver>(m, class_name.c_str())
      .def(py::init<TRunner&, TPile&, Store&, U, U, bool, bool,
                    Const<TF> const*, ConstiN const*, bool>(),
           py::arg("runner"), py::arg("pile"), py::arg("store"),
           py::arg("max_num_particles"), py::arg("num_ushers") = 0,
           py::arg("enable_surface_tension") = false,
           py::arg("enable_vorticity") = false, py::arg("cn") = nullptr,
           py::arg("cni") = nullptr, py::arg("graphical") = false)
      .def_readonly("num_density_solve", &TSolverI::num_density_solve)
      .def_readonly("mean_density_error", &TSolverI::mean_density_error)
      .def_readwrite("density_error_tolerance",
                     &TSolverI::density_error_tolerance)
      .def_readwrite("min_density_solve", &TSolverI::min_density_solve)
      .def_readwrite("max_density_solve", &TSolverI::max_density_solve)
      .def_property_readonly(
          "particle_guiding",
          [](TSolverI const& solver) { return solver.particle_guiding.get(); })
      .def_property_readonly(
          "particle_pressure",
          [](TSolverI const& solver) { return solver.particle_pressure.get(); })
      .def_property_readonly("particle_last_pressure",
                             [](TSolverI const& solver) {
                               return solver.particle_last_pressure.get();
                             })
      .def_property_readonly("particle_diag_adv_density",
                             [](TSolverI const& solver) {
                               return solver.particle_diag_adv_density.get();
                             })
      .def_property_readonly("particle_pressure_accel",
                             [](TSolverI const& solver) {
                               return solver.particle_pressure_accel.get();
                             })
      .def_property_readonly("particle_density_err",
                             [](TSolverI const& solver) {
                               return solver.particle_density_err.get();
                             })
      .def("step", &TSolverI::template step<0>)
      .def("step_wrap1", &TSolverI::template step<1>);
}

template <typename TF>
void declare_solver_pellet(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TSolver = Solver<TF>;
  using TSolverPellet = SolverPellet<TF>;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  std::string class_name = std::string("SolverPellet") + name;
  py::class_<TSolverPellet, TSolver>(m, class_name.c_str())
      .def(py::init<TRunner&, TPile&, Store&, U, Const<TF> const*,
                    ConstiN const*, bool>(),
           py::arg("runner"), py::arg("pile"), py::arg("store"),
           py::arg("max_num_particles"), py::arg("cn") = nullptr,
           py::arg("cni") = nullptr, py::arg("graphical") = false)
      .def_readwrite("cfl_factor", &TSolverPellet::cfl_factor)
      .def_readwrite("cohesion", &TSolverPellet::cohesion)
      .def_readwrite("adhesion", &TSolverPellet::adhesion)
      .def_property_readonly("particle_dfsph_factor",
                             [](TSolverPellet const& solver) {
                               return solver.particle_dfsph_factor.get();
                             })
      .def("set_pellets", &TSolverPellet::set_pellets)
      .def("step", &TSolverPellet::template step<0>)
      .def("step_wrap1", &TSolverPellet::template step<1>);
}

template <typename TF>
void declare_usher(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  using TUsher = Usher<TF>;
  using TPile = Pile<TF>;
  std::string class_name = std::string("Usher") + name;
  py::class_<TUsher>(m, class_name.c_str())
      .def(py::init<Store&, TPile&, U>(), py::arg("store"), py::arg("pile"),
           py::arg("num_ushers"))
      .def_readwrite("num_ushers", &TUsher::num_ushers)
      .def_property_readonly(
          "focal_x", [](TUsher const& usher) { return usher.focal_x.get(); })
      .def_property_readonly(
          "focal_v", [](TUsher const& usher) { return usher.focal_v.get(); })
      .def_property_readonly(
          "direction",
          [](TUsher const& usher) { return usher.direction.get(); })
      .def_property_readonly(
          "usher_kernel_radius",
          [](TUsher const& usher) { return usher.usher_kernel_radius.get(); })
      .def_property_readonly(
          "drive_strength",
          [](TUsher const& usher) { return usher.drive_strength.get(); })
      .def("reset", &TUsher::reset);
}

template <typename TF>
void declare_display_proxy(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TDisplayProxy = DisplayProxy<TF>;
  using TSolverIi = SolverIi<TF>;
  using TSolverDf = SolverDf<TF>;
  std::string class_name = std::string("DisplayProxy") + name;
  py::class_<TDisplayProxy>(m, class_name.c_str())
      .def("create_colormap_viridis", &TDisplayProxy::create_colormap_viridis)
      .def("add_particle_shading_program",
           &TDisplayProxy::add_particle_shading_program, py::arg("x"),
           py::arg("attr"), py::arg("colormap_tex"), py::arg("particle_radius"),
           py::arg("solver"), py::arg("clear") = true)
      .def("add_pile_shading_program", &TDisplayProxy::add_pile_shading_program)
      .def("add_map_graphical_pointers",
           &TDisplayProxy::add_map_graphical_pointers)
      .def("add_unmap_graphical_pointers",
           &TDisplayProxy::add_unmap_graphical_pointers)
      .def("add_normalize",
           &TDisplayProxy::template add_normalize<Variable<1, TF3>>)
      .def("add_normalize",
           &TDisplayProxy::template add_normalize<Variable<1, TF>>)
      .def("add_step", &TDisplayProxy::template add_step<TSolverDf>)
      .def("add_step", &TDisplayProxy::template add_step<TSolverIi>)
      .def("add_clear", &TDisplayProxy::add_clear)
      .def("run", &TDisplayProxy::run)
      .def("draw", &TDisplayProxy::draw)
      .def("set_camera", &TDisplayProxy::set_camera)
      .def("set_clip_planes", &TDisplayProxy::set_clip_planes)
      .def("create_framebuffer", &TDisplayProxy::create_framebuffer,
           py::return_value_policy::reference)
      .def("bind_framebuffer", &TDisplayProxy::bind_framebuffer)
      .def("add_bind_framebuffer_step",
           &TDisplayProxy::add_bind_framebuffer_step)
      .def("add_show_framebuffer_shader",
           &TDisplayProxy::add_show_framebuffer_shader)
      .def("resize", &TDisplayProxy::resize);
}

template <typename TF>
py::class_<Runner<TF>> declare_runner(py::module& m, const char* name) {
  using TRunner = Runner<TF>;
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  std::string class_name = std::string("Runner") + name;
  return py::class_<TRunner>(m, class_name.c_str())
      .def(py::init<>())
      .def_readonly("launch_stat_dict", &TRunner::launch_stat_dict_)
      .def_readonly("custom_elapsed_dict", &TRunner::custom_elapsed_dict_)
      .def("launch_print_cn", &TRunner::launch_print_cn)
      .def("launch_create_fluid_block", &TRunner::launch_create_fluid_block,
           py::arg("particle_x"), py::arg("num_particles"), py::arg("offset"),
           py::arg("particle_radius"), py::arg("mode"), py::arg("box_min"),
           py::arg("box_max"))
      .def("launch_create_custom_beads_internal",
           &TRunner::launch_create_custom_beads_internal, py::arg("particle_x"),
           py::arg("ref_x"), py::arg("internal_encoded_sorted"),
           py::arg("num_particles"), py::arg("offset"))
      .def("launch_create_custom_beads_scalar_internal",
           &TRunner::launch_create_custom_beads_scalar_internal,
           py::arg("particle_scalar"), py::arg("ref_scalar"),
           py::arg("internal_encoded_sorted"), py::arg("num_particles"),
           py::arg("offset"))
      .def("launch_create_fluid_block_internal",
           &TRunner::launch_create_fluid_block_internal, py::arg("particle_x"),
           py::arg("internal_encoded_sorted"), py::arg("num_particles"),
           py::arg("offset"), py::arg("particle_radius"), py::arg("mode"),
           py::arg("box_min"), py::arg("box_max"))
      .def("launch_create_fluid_cylinder",
           &TRunner::launch_create_fluid_cylinder, py::arg("particle_x"),
           py::arg("num_particles"), py::arg("offset"), py::arg("radius"),
           py::arg("particle_radius"), py::arg("y_min"), py::arg("y_max"))
      .def("launch_create_fluid_cylinder_internal",
           &TRunner::launch_create_fluid_cylinder_internal,
           py::arg("particle_x"), py::arg("internal_encoded_sorted"),
           py::arg("num_particles"), py::arg("offset"), py::arg("radius"),
           py::arg("particle_radius"), py::arg("y_min"), py::arg("y_max"))
      .def("launch_compute_particle_boundary",
           &TRunner::launch_compute_particle_boundary)
      .def("launch_compute_particle_boundary_with_pellets",
           &TRunner::launch_compute_particle_boundary_with_pellets,
           py::arg("particle_boundary_kernel_combined"),
           py::arg("sample_pellet_neighbors"),
           py::arg("sample_num_pellet_neighbors"), py::arg("num_particles"))
      .def("launch_compute_density_mask", &TRunner::launch_compute_density_mask)
      .def("launch_compute_boundary_mask",
           &TRunner::launch_compute_boundary_mask)
      .def("launch_update_particle_grid", &TRunner::launch_update_particle_grid)
      .def("launch_make_neighbor_list",
           &TRunner::template launch_make_neighbor_list<0>)
      .def("launch_make_neighbor_list_wrap1",
           &TRunner::template launch_make_neighbor_list<1>)
      .def("launch_make_bead_pellet_neighbor_list",
           &TRunner::template launch_make_bead_pellet_neighbor_list<0>,
           py::arg("sample_x"), py::arg("pid"), py::arg("pid_length"),
           py::arg("sample_bead_neighbors"),
           py::arg("sample_num_bead_neighbors"),
           py::arg("sample_pellet_neighbors"),
           py::arg("sample_num_pellet_neighbors"), py::arg("grid_anomaly"),
           py::arg("max_num_beads"), py::arg("num_samples"),
           py::arg("offset") = 0)
      .def("launch_make_bead_pellet_neighbor_list_wrap1",
           &TRunner::template launch_make_bead_pellet_neighbor_list<1>,
           py::arg("sample_x"), py::arg("pid"), py::arg("pid_length"),
           py::arg("sample_bead_neighbors"),
           py::arg("sample_num_bead_neighbors"),
           py::arg("sample_pellet_neighbors"),
           py::arg("sample_num_pellet_neighbors"), py::arg("grid_anomaly"),
           py::arg("max_num_beads"), py::arg("num_samples"),
           py::arg("offset") = 0)
      .def("launch_compute_density", &TRunner::launch_compute_density)
      .def("launch_compute_density_with_pellets",
           &TRunner::launch_compute_density_with_pellets,
           py::arg("particle_neighbors"), py::arg("particle_num_neighbors"),
           py::arg("particle_boundary_neighbors"),
           py::arg("particle_num_boundary_neighbors"),
           py::arg("particle_density"), py::arg("num_particles"))
      .def("launch_sample_fluid", &TRunner::template launch_sample_fluid<TF>)
      .def("launch_sample_fluid", &TRunner::template launch_sample_fluid<TF3>)
      .def("launch_sample_fluid_density", &TRunner::launch_sample_fluid_density)
      .def("launch_sample_velocity", &TRunner::launch_sample_velocity)
      .def("launch_sample_velocity_with_pellets",
           &TRunner::launch_sample_velocity_with_pellets, py::arg("sample_x"),
           py::arg("particle_x"), py::arg("particle_density"),
           py::arg("particle_v"), py::arg("sample_neighbors"),
           py::arg("sample_num_neighbors"), py::arg("sample_v"),
           py::arg("sample_pellet_neighbors"),
           py::arg("sample_num_pellet_neighbors"), py::arg("num_samples"))
      .def("launch_compute_distance_mask_multiple",
           &TRunner::launch_compute_distance_mask_multiple, py::arg("grid_x"),
           py::arg("buoy_x"), py::arg("mask"), py::arg("distance_threshold"),
           py::arg("num_grid_points"), py::arg("num_buoys"))
      .def("launch_sample_vorticity", &TRunner::launch_sample_vorticity)
      .def("launch_sample_vorticity_with_pellets",
           &TRunner::launch_sample_vorticity_with_pellets)
      .def("launch_sample_density", &TRunner::launch_sample_density)
      .def("launch_sample_density_with_pellets",
           &TRunner::launch_sample_density_with_pellets)
      .def("launch_histogram256", &TRunner::launch_histogram256)
      .def("launch_histogram256_with_mask",
           &TRunner::launch_histogram256_with_mask)
      .def_static("get_fluid_block_num_particles",
                  &TRunner::get_fluid_block_num_particles, py::arg("mode"),
                  py::arg("box_min"), py::arg("box_max"),
                  py::arg("particle_radius"))
      .def_static("get_fluid_cylinder_num_particles",
                  &TRunner::get_fluid_cylinder_num_particles, py::arg("radius"),
                  py::arg("y_min"), py::arg("y_max"),
                  py::arg("particle_radius"))
      .def_static("sqrt_inplace", &TRunner::template sqrt_inplace<1, TF>,
                  py::arg("var"), py::arg("n"), py::arg("offset") = 0)
      .def_static("norm", &TRunner::template norm<1>, py::arg("v"),
                  py::arg("s"), py::arg("n"), py::arg("offset") = 0)
      .def_static("norm_xz", &TRunner::template norm_xz<1>, py::arg("v"),
                  py::arg("s"), py::arg("n"), py::arg("offset") = 0)
      .def_static("extract_x", &TRunner::template extract_x<1>, py::arg("v"),
                  py::arg("s"), py::arg("n"), py::arg("offset") = 0)
      .def_static("extract_y", &TRunner::template extract_y<1>, py::arg("v"),
                  py::arg("s"), py::arg("n"), py::arg("offset") = 0)
      .def_static("extract_z", &TRunner::template extract_z<1>, py::arg("v"),
                  py::arg("s"), py::arg("n"), py::arg("offset") = 0)
      .def_static(
          "max_inplace",
          py::overload_cast<Variable<1, TF>&, Variable<1, TF> const&, U, U>(
              &TRunner::template max_inplace<1, TF>),
          py::arg("target"), py::arg("other"), py::arg("n"),
          py::arg("offset") = 0)
      .def_static("max_inplace",
                  py::overload_cast<Variable<1, TF>&, Variable<1, TF> const&>(
                      &TRunner::template max_inplace<1, TF>),
                  py::arg("target"), py::arg("other"))
      .def_static("min", &TRunner::template min<1, TF>, py::arg("var"),
                  py::arg("n"), py::arg("offset") = 0)
      .def_static("max", &TRunner::template max<1, TF>, py::arg("var"),
                  py::arg("n"), py::arg("offset") = 0);
}

PYBIND11_MODULE(_alluvion, m) {
  m.doc() = "CUDA backend for SPH fluid simulation.";
  using namespace pybind11;

  py::class_<Store> store_class =
      py::class_<Store>(m, "Store")
          .def(py::init<>())
          .def_static("set_device", &Store::set_device)
          .def_static("get_alu_info", &Store::get_alu_info)
          .def("has_display", &Store::has_display)
          .def("create_display",
               [](Store& store, int width, int height, const char* title,
                  bool offscreen) -> void {
                 store.create_display(width, height, title, offscreen);
               })
          .def("get_display_proxyfloat",
               [](Store& store) {
                 return DisplayProxy<float>(store.get_display());
               })
          .def("get_display_proxydouble",
               [](Store& store) {
                 return DisplayProxy<double>(store.get_display());
               })
          .def_property_readonly("cnfloat", &Store::template get_cn<float>)
          .def_property_readonly("cndouble", &Store::template get_cn<double>)
          .def_property_readonly("cni", &Store::template get_cni)
          .def("create_cnfloat", &Store::template create_cn<float>)
          .def("create_cndouble", &Store::template create_cn<double>)
          .def("copy_cn_externalfloat",
               &Store::template copy_cn_external<float>)
          .def("copy_cn_externaldouble",
               &Store::template copy_cn_external<double>)
          .def("copy_cnfloat", &Store::template copy_cn<float>)
          .def("copy_cndouble", &Store::template copy_cn<double>)
          .def("map_graphical_pointers", &Store::map_graphical_pointers)
          .def("unmap_graphical_pointers", &Store::unmap_graphical_pointers);

  py::class_<Runner<float>> runner_float = declare_runner<float>(m, "float");
  py::class_<Runner<double>> runner_double =
      declare_runner<double>(m, "double");

  declare_variable<1, float, bool>(m, store_class, &runner_float, nullptr,
                                   "1Dfloat");
  declare_variable<1, float2, float>(m, store_class, &runner_float, nullptr,
                                     "1Dfloat2");
  declare_variable<1, float3, float>(m, store_class, &runner_float, nullptr,
                                     "1Dfloat3");
  declare_variable<1, float4, float>(m, store_class, &runner_float, nullptr,
                                     "1Dfloat4");
  declare_variable<2, float, bool>(m, store_class, &runner_float, nullptr,
                                   "2Dfloat");
  declare_variable<2, float3, float>(m, store_class, &runner_float, nullptr,
                                     "2Dfloat3");
  declare_variable<2, float4, float>(m, store_class, &runner_float, nullptr,
                                     "2Dfloat4");
  declare_variable<4, float4, float>(m, store_class, &runner_float, nullptr,
                                     "4Dfloat4");

  declare_variable<1, double, bool>(m, store_class, nullptr, &runner_double,
                                    "1Ddouble");
  declare_variable<1, double2, double>(m, store_class, nullptr, &runner_double,
                                       "1Ddouble2");
  declare_variable<1, double3, double>(m, store_class, nullptr, &runner_double,
                                       "1Ddouble3");
  declare_variable<1, double4, double>(m, store_class, nullptr, &runner_double,
                                       "1Ddouble4");
  declare_variable<2, double, bool>(m, store_class, nullptr, &runner_double,
                                    "2Ddouble");
  declare_variable<2, double3, double>(m, store_class, nullptr, &runner_double,
                                       "2Ddouble3");
  declare_variable<2, double4, double>(m, store_class, nullptr, &runner_double,
                                       "2Ddouble4");
  declare_variable<4, double4, double>(m, store_class, nullptr, &runner_double,
                                       "4Ddouble4");

  declare_variable<1, uint, bool>(m, store_class, &runner_float, &runner_double,
                                  "1Duint");
  declare_variable<3, uint, bool>(m, store_class, &runner_float, &runner_double,
                                  "3Duint");

  declare_metrics<1, float3>(m, &runner_float);
  declare_metrics<1, double3>(m, &runner_double);

  declare_pinned_variable<1, float3>(m, "1Dfloat3");
  declare_pinned_variable<1, double3>(m, "1Ddouble3");
  declare_pinned_variable<1, float4>(m, "1Dfloat4");
  declare_pinned_variable<1, double4>(m, "1Ddouble4");

  py::enum_<NumericType>(m, "NumericType")
      .value("f32", NumericType::f32)
      .value("f64", NumericType::f64)
      .value("i32", NumericType::i32)
      .value("u32", NumericType::u32)
      .value("undefined", NumericType::undefined);

  declare_vector2<float2, float>(m, "float2");
  declare_vector2<double2, double>(m, "double2");

  declare_vector3<float3, float>(m, "float3");
  declare_vector3<double3, double>(m, "double3");
  declare_vector3<int3, int>(m, "int3");
  declare_vector3<uint3, uint>(m, "uint3");

  declare_vector4<float4, float>(m, "float4");
  declare_vector4<double4, double>(m, "double4");

  // Mesh should be declared before Pile to allow Mesh() as default arguments
  py::class_<Mesh>(m, "Mesh")
      .def_readonly("vertices", &Mesh::vertices)
      .def_readonly("normals", &Mesh::normals)
      .def_readonly("texcoords", &Mesh::texcoords)
      .def_readonly("faces", &Mesh::faces)
      .def(py::init<>())
      .def("set_box", &Mesh::set_box)
      .def("set_uv_sphere", &Mesh::set_uv_sphere)
      .def("set_cylinder", &Mesh::set_cylinder)
      .def("set_obj", &Mesh::set_obj)
      .def("calculate_normals", &Mesh::calculate_normals)
      .def("translate", &Mesh::translate)
      .def("rotate", &Mesh::rotate)
      .def("scale", &Mesh::scale)
      .def("clear", &Mesh::clear)
      .def("export_obj", &Mesh::export_obj)
      .def(
          "calculate_mass_properties",
          [](Mesh const& mesh, float density) {
            float3 com, inertia_diag, inertia_off_diag;
            float mass = mesh.calculate_mass_properties(
                com, inertia_diag, inertia_off_diag, density);
            return std::make_tuple(mass, com, inertia_diag, inertia_off_diag);
          },
          py::arg("density") = 1)
      .def("copy_to", &Mesh::copy_to<float>)
      .def("copy_to", &Mesh::copy_to<double>);

  py::class_<CompleteFramebuffer>(m, "CompleteFramebuffer")
      .def_readonly("width", &CompleteFramebuffer::width_)
      .def_readonly("height", &CompleteFramebuffer::height_)
      .def("write", &CompleteFramebuffer::write)
      .def("get", &CompleteFramebuffer::get)
      .def("read", &CompleteFramebuffer::read);
  py::enum_<VolumeMethod>(m, "VolumeMethod")
      .value("volume_map", VolumeMethod::volume_map)
      .value("pellets", VolumeMethod::pellets);
  declare_pile<float>(m, "float");
  declare_pile<double>(m, "double");

  declare_solver<float>(m, "float");
  declare_solver<double>(m, "double");

  declare_solver_df<float>(m, "float");
  declare_solver_df<double>(m, "double");

  declare_solver_ii<float>(m, "float");
  declare_solver_ii<double>(m, "double");

  declare_solver_i<float>(m, "float");
  declare_solver_i<double>(m, "double");

  declare_solver_pellet<float>(m, "float");
  declare_solver_pellet<double>(m, "double");

  declare_usher<float>(m, "float");
  declare_usher<double>(m, "double");

  declare_display_proxy<float>(m, "float");
  declare_display_proxy<double>(m, "double");

  declare_const<float>(m, "float");
  declare_const<double>(m, "double");

  py::class_<ConstiN>(m, "ConstiN")
      .def_readonly("max_num_contacts", &ConstiN::max_num_contacts)
      .def_readwrite("num_boundaries", &ConstiN::num_boundaries)
      .def_readwrite("max_num_particles_per_cell",
                     &ConstiN::max_num_particles_per_cell)
      .def_readwrite("grid_res", &ConstiN::grid_res)
      .def_readwrite("grid_offset", &ConstiN::grid_offset)
      .def_readwrite("max_num_neighbors_per_particle",
                     &ConstiN::max_num_neighbors_per_particle);

  py::module m_dg = m.def_submodule("dg", "Discregrid");

  declare_distance<float3, float>(m_dg, "float");
  declare_distance<double3, double>(m_dg, "double");

  declare_sphere_distance<float3, float>(m_dg, "float");
  declare_sphere_distance<double3, double>(m_dg, "double");
  declare_box_distance<float3, float>(m_dg, "float");
  declare_box_distance<double3, double>(m_dg, "double");
  declare_box_shell_distance<float3, float>(m_dg, "float");
  declare_box_shell_distance<double3, double>(m_dg, "double");
  declare_cylinder_distance<float3, float>(m_dg, "float");
  declare_cylinder_distance<double3, double>(m_dg, "double");
  declare_infinite_cylinder_distance<float3, float>(m_dg, "float");
  declare_infinite_cylinder_distance<double3, double>(m_dg, "double");
  declare_infinite_tube_distance<float3, float>(m_dg, "float");
  declare_infinite_tube_distance<double3, double>(m_dg, "double");
  declare_capsule_distance<float3, float>(m_dg, "float");
  declare_capsule_distance<double3, double>(m_dg, "double");
  declare_mesh_distance<float3, float>(m_dg, "float");
  declare_mesh_distance<double3, double>(m_dg, "double");
  declare_triangle_mesh<float>(m_dg, "float");
  declare_triangle_mesh<double>(m_dg, "double");

  m.attr("kHistogram256BinCount") = py::int_(kHistogram256BinCount);
  m.attr("kPartialHistogram256Size") = py::int_(kPartialHistogram256Size);
}
