#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "alluvion/display.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;
namespace py = pybind11;

template <unsigned int D, typename M>
void declare_variable(py::module& m, py::class_<Store>& store_class,
                      const char* name) {
  using Class = Variable<D, M>;
  std::string create_func_name = std::string("create") + name;
  store_class.def(create_func_name.c_str(), &Store::create<D, M>);

  std::string variable_name = std::string("Variable") + name;
  py::class_<Class>(m, variable_name.c_str())
      .def(py::init<const Class&>())
      .def("get_bytes",
           [](Class& variable, py::array_t<unsigned char> bytes) {
             variable.get_bytes(bytes.mutable_data(), bytes.size());
           })
      .def("set_bytes",
           [](Class& variable, py::array_t<unsigned char> bytes) {
             variable.set_bytes(bytes.data(), bytes.size());
           })
      .def("get_type", &Class::get_type)
      .def("get_num_primitives_per_unit", &Class::get_num_primitives_per_unit)
      .def("get_linear_shape", &Class::get_linear_shape)
      .def("get_num_primitives", &Class::get_num_primitives)
      .def("get_shape", &Class::get_shape);
}

PYBIND11_MODULE(_alluvion, m) {
  m.doc() = "CUDA backend for SPH fluid simulation.";
  using namespace pybind11;

  py::class_<Store> store_class =
      py::class_<Store>(m, "Store").def(py::init<>());

  declare_variable<1, F>(m, store_class, "1DF");
  declare_variable<1, F3>(m, store_class, "1DF3");
  declare_variable<2, F3>(m, store_class, "2DF3");

  // py::class_<Runner>(m, "Runner")
  //     .def_static(
  //         "sum", [](py::array_t<unsigned char> dst, Variable<D, F3>& var)
  //         {
  //           Runner::sum<F>(reinterpret_cast<void*>(dst.mutable_data()),
  //                          var.ptr_, var.get_num_primitives(), var.type_);
  //         });
  py::enum_<NumericType>(m, "NumericType")
      .value("f32", NumericType::f32)
      .value("f64", NumericType::f64)
      .value("i32", NumericType::i32)
      .value("u32", NumericType::u32)
      .value("undefined", NumericType::undefined);
  py::class_<Display>(m, "Display").def(py::init<int, int, const char*>());
}
