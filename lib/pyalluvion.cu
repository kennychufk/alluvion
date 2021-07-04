#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "alluvion/constants.hpp"
#include "alluvion/dg/box_distance.hpp"
#include "alluvion/dg/cylinder_distance.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/display.hpp"
#include "alluvion/display_proxy.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/solver_df.hpp"
#include "alluvion/solver_ii.hpp"
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

template <typename TF3, typename TF>
void declare_vector3(py::module& m, const char* name) {
  py::class_<TF3>(m, name)
      .def(py::init<TF, TF, TF>())
      .def_readwrite("x", &TF3::x)
      .def_readwrite("y", &TF3::y)
      .def_readwrite("z", &TF3::z)
      .def("__repr__", [](TF3 const& v) {
        std::stringstream stream;
        stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return stream.str();
      });
}

template <typename TF4, typename TF>
void declare_vector4(py::module& m, const char* name) {
  py::class_<TF4>(m, name)
      .def(py::init<TF, TF, TF, TF>())
      .def_readwrite("x", &TF4::x)
      .def_readwrite("y", &TF4::y)
      .def_readwrite("z", &TF4::z)
      .def("__repr__", [](TF4 const& v) {
        std::stringstream stream;
        stream << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w
               << ")";
        return stream.str();
      });
}

template <unsigned int D, typename M>
void declare_variable_with_store_creation(py::module& m,
                                          py::class_<Store>& store_class,
                                          const char* name) {
  using VariableClass = Variable<D, M>;
  using GraphicalVariableClass = GraphicalVariable<D, M>;
  std::string create_func_name = std::string("create") + name;
  store_class.def(create_func_name.c_str(), &Store::create<D, M>,
                  py::return_value_policy::take_ownership);
  std::string create_graphical_func_name =
      std::string("create_graphical") + name;
  store_class.def(create_graphical_func_name.c_str(),
                  &Store::create_graphical<D, M>,
                  py::return_value_policy::take_ownership);

  std::string variable_name = std::string("Variable") + name;
  py::class_<VariableClass>(m, variable_name.c_str())
      .def(py::init<const VariableClass&>())
      .def_readonly("ptr", &VariableClass::ptr_)
      .def("get_bytes",
           [](VariableClass& variable, py::array_t<unsigned char> bytes) {
             variable.get_bytes(bytes.mutable_data(), bytes.size());
           })
      .def("set_bytes",
           [](VariableClass& variable, py::array_t<unsigned char> bytes) {
             variable.set_bytes(bytes.data(), bytes.size());
           })
      .def("get_type", &VariableClass::get_type)
      .def("get_num_primitives_per_unit",
           &VariableClass::get_num_primitives_per_unit)
      .def("get_linear_shape", &VariableClass::get_linear_shape)
      .def("get_num_primitives", &VariableClass::get_num_primitives)
      .def("get_shape", &VariableClass::get_shape);

  std::string graphical_variable_name = std::string("GraphicalVariable") + name;
  py::class_<GraphicalVariableClass, VariableClass>(
      m, graphical_variable_name.c_str())
      .def(py::init<const GraphicalVariableClass&>())
      .def_readonly("vbo", &GraphicalVariableClass::vbo_);
}

template <typename TF>
void declare_pile(py::module& m, const char* name) {
  using TPile = Pile<TF>;
  std::string class_name = std::string("Pile") + name;
  py::class_<TPile>(m, class_name.c_str())
      .def(py::init<Store&, U>())
      .def_readwrite("mass", &TPile::mass_)
      .def_readwrite("restitution", &TPile::restitution_)
      .def_readwrite("friction", &TPile::friction_)
      .def_readwrite("inertia_tensor", &TPile::inertia_tensor_)
      .def_property_readonly("distance_grids", &TPile::get_distance_grids)
      .def_property_readonly("volume_grids", &TPile::get_volume_grids)
      .def("add", &TPile::add, py::arg("distance"), py::arg("resolution"),
           py::arg("sign"), py::arg("thickness"), py::arg("collision_mesh"),
           py::arg("mass"), py::arg("restitution"), py::arg("friction"),
           py::arg("inertia_tensor"), py::arg("x"), py::arg("q"),
           py::arg("display_mesh"))
      .def("replace", &TPile::replace, py::arg("i"), py::arg("distance"),
           py::arg("resolution"), py::arg("sign"), py::arg("thickness"),
           py::arg("collision_mesh"), py::arg("mass"), py::arg("restitution"),
           py::arg("friction"), py::arg("inertia_tensor"), py::arg("x"),
           py::arg("q"), py::arg("display_mesh"))
      .def("build_grids", &TPile::build_grids)
      .def("set_gravity", &TPile::set_gravity)
      .def("reallocate_kinematics_on_device",
           &TPile::reallocate_kinematics_on_device)
      .def("reallocate_kinematics_on_pinned",
           &TPile::reallocate_kinematics_on_pinned)
      .def("copy_kinematics_to_device", &TPile::copy_kinematics_to_device)
      .def("integrate_kinematics", &TPile::integrate_kinematics)
      .def("calculate_cfl_v2", &TPile::calculate_cfl_v2)
      .def("find_contacts", &TPile::find_contacts)
      .def("solve_contacts", &TPile::solve_contacts)
      .def("get_size", &TPile::get_size)
      .def("get_matrix", &TPile::get_matrix);
}

template <typename TF3, typename TF>
void declare_distance(py::module& m, const char* name) {
  using TDistance = dg::Distance<TF3, TF>;
  std::string class_name = std::string("Distance") + name;
  py::class_<TDistance, dg::PyDistance<TF3, TF>>(m, class_name.c_str())
      .def("get_aabb_min",
           [](TDistance const& distance) -> std::array<TF, 3> {
             TF3 aabb_min = distance.get_aabb_min();
             return {aabb_min.x, aabb_min.y, aabb_min.z};
           })
      .def("get_aabb_max",
           [](TDistance const& distance) -> std::array<TF, 3> {
             TF3 aabb_max = distance.get_aabb_max();
             return {aabb_max.x, aabb_max.y, aabb_max.z};
           })
      .def("get_max_distance", &TDistance::get_max_distance);
}

template <typename TF3, typename TF>
void declare_sphere_distance(py::module& m, const char* name) {
  using TSphereDistance = dg::SphereDistance<TF3, TF>;
  std::string class_name = std::string("SphereDistance") + name;
  py::class_<TSphereDistance, dg::Distance<TF3, TF>>(m, class_name.c_str())
      .def(py::init<TF>());
}

template <typename TF3, typename TF>
void declare_box_distance(py::module& m, const char* name) {
  using TBoxDistance = dg::BoxDistance<TF3, TF>;
  std::string class_name = std::string("BoxDistance") + name;
  py::class_<TBoxDistance, dg::Distance<TF3, TF>>(m, class_name.c_str())
      .def(py::init<TF3>());
}

template <typename TF3, typename TF>
void declare_cylinder_distance(py::module& m, const char* name) {
  using TCylinderDistance = dg::CylinderDistance<TF3, TF>;
  std::string class_name = std::string("CylinderDistance") + name;
  py::class_<TCylinderDistance, dg::Distance<TF3, TF>>(m, class_name.c_str())
      .def(py::init<TF, TF, TF>());
}

template <typename TF>
void declare_const(py::module& m, const char* name) {
  using TConst = Const<TF>;
  std::string class_name = std::string("Const") + name;
  py::class_<TConst>(m, class_name.c_str())
      .def("set_cubic_discretization_constants",
           &TConst::set_cubic_discretization_constants)
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
      .def_readwrite("gravity", &TConst::gravity)
      .def_readwrite("axial_gravity", &TConst::axial_gravity)
      .def_readwrite("radial_gravity", &TConst::radial_gravity)
      .def_readwrite("boundary_epsilon", &TConst::boundary_epsilon)
      .def_readwrite("dfsph_factor_epsilon", &TConst::dfsph_factor_epsilon)
      .def_readwrite("contact_tolerance", &TConst::contact_tolerance);
}

template <typename TF>
void declare_solver(py::module& m, const char* name) {
  using TSolver = Solver<TF>;
  std::string class_name = std::string("Solver") + name;
  py::class_<TSolver>(m, class_name.c_str())
      .def_readwrite("num_particles", &TSolver::num_particles)
      .def_readwrite("dt", &TSolver::dt)
      .def_readwrite("max_dt", &TSolver::max_dt)
      .def_readwrite("min_dt", &TSolver::min_dt)
      .def_readwrite("cfl", &TSolver::cfl)
      .def_readwrite("particle_radius", &TSolver::particle_radius);
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
      .def(py::init<TRunner&, TPile&, Store&, U, U3, U, U, bool>())
      .def_property_readonly(
          "particle_x",
          [](TSolverDf const& solver) { return solver.particle_x.get(); })
      .def_property_readonly(
          "particle_v",
          [](TSolverDf const& solver) { return solver.particle_v.get(); })
      .def_property_readonly(
          "particle_a",
          [](TSolverDf const& solver) { return solver.particle_a.get(); })
      .def_property_readonly(
          "particle_density",
          [](TSolverDf const& solver) { return solver.particle_density.get(); })
      .def_property_readonly("particle_boundary_xj",
                             [](TSolverDf const& solver) {
                               return solver.particle_boundary_xj.get();
                             })
      .def_property_readonly("particle_boundary_volume",
                             [](TSolverDf const& solver) {
                               return solver.particle_boundary_volume.get();
                             })
      .def_property_readonly(
          "particle_force",
          [](TSolverDf const& solver) { return solver.particle_force.get(); })
      .def_property_readonly(
          "particle_torque",
          [](TSolverDf const& solver) { return solver.particle_torque.get(); })
      .def_property_readonly(
          "particle_cfl_v2",
          [](TSolverDf const& solver) { return solver.particle_cfl_v2.get(); })
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
      .def_property_readonly(
          "pid", [](TSolverDf const& solver) { return solver.pid.get(); })
      .def_property_readonly(
          "pid_length",
          [](TSolverDf const& solver) { return solver.pid_length.get(); })
      .def_property_readonly("particle_neighbors",
                             [](TSolverDf const& solver) {
                               return solver.particle_neighbors.get();
                             })
      .def_property_readonly("particle_num_neighbors",
                             [](TSolverDf const& solver) {
                               return solver.particle_num_neighbors.get();
                             })
      .def("step", &TSolverDf::template step<0, 0>);
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
      .def(py::init<TRunner&, TPile&, Store&, U, U3, U, U, bool>())
      .def_property_readonly(
          "particle_x",
          [](TSolverIi const& solver) { return solver.particle_x.get(); })
      .def_property_readonly(
          "particle_v",
          [](TSolverIi const& solver) { return solver.particle_v.get(); })
      .def_property_readonly(
          "particle_a",
          [](TSolverIi const& solver) { return solver.particle_a.get(); })
      .def_property_readonly(
          "particle_density",
          [](TSolverIi const& solver) { return solver.particle_density.get(); })
      .def_property_readonly("particle_boundary_xj",
                             [](TSolverIi const& solver) {
                               return solver.particle_boundary_xj.get();
                             })
      .def_property_readonly("particle_boundary_volume",
                             [](TSolverIi const& solver) {
                               return solver.particle_boundary_volume.get();
                             })
      .def_property_readonly(
          "particle_force",
          [](TSolverIi const& solver) { return solver.particle_force.get(); })
      .def_property_readonly(
          "particle_torque",
          [](TSolverIi const& solver) { return solver.particle_torque.get(); })
      .def_property_readonly(
          "particle_cfl_v2",
          [](TSolverIi const& solver) { return solver.particle_cfl_v2.get(); })
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
      .def_property_readonly(
          "particle_sum_tmp",
          [](TSolverIi const& solver) { return solver.particle_sum_tmp.get(); })
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
      .def_property_readonly(
          "pid", [](TSolverIi const& solver) { return solver.pid.get(); })
      .def_property_readonly(
          "pid_length",
          [](TSolverIi const& solver) { return solver.pid_length.get(); })
      .def_property_readonly("particle_neighbors",
                             [](TSolverIi const& solver) {
                               return solver.particle_neighbors.get();
                             })
      .def_property_readonly("particle_num_neighbors",
                             [](TSolverIi const& solver) {
                               return solver.particle_num_neighbors.get();
                             })
      .def("step", &TSolverIi::template step<0, 0>);
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
           &TDisplayProxy::add_particle_shading_program)
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
      .def("run", &TDisplayProxy::run)
      .def("set_camera", &TDisplayProxy::set_camera);
}

template <typename TF>
void declare_runner(py::module& m, const char* name) {
  using TRunner = Runner<TF>;
  std::string class_name = std::string("Runner") + name;
  py::class_<TRunner>(m, class_name.c_str())
      .def(py::init<>())
      .def("launch_create_fluid_block", &TRunner::launch_create_fluid_block)
      .def("launch_create_fluid_cylinder",
           &TRunner::launch_create_fluid_cylinder);
}

PYBIND11_MODULE(_alluvion, m) {
  m.doc() = "CUDA backend for SPH fluid simulation.";
  using namespace pybind11;

  py::class_<Store> store_class =
      py::class_<Store>(m, "Store")
          .def(py::init<>())
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
          .def("copy_cnfloat", &Store::template copy_cn<float>)
          .def("copy_cndouble", &Store::template copy_cn<double>)
          .def("map_graphical_pointers", &Store::map_graphical_pointers)
          .def("unmap_graphical_pointers", &Store::unmap_graphical_pointers);

  declare_variable_with_store_creation<1, float>(m, store_class, "1Dfloat");
  declare_variable_with_store_creation<1, float3>(m, store_class, "1Dfloat3");
  declare_variable_with_store_creation<2, float>(m, store_class, "2Dfloat");
  declare_variable_with_store_creation<2, float3>(m, store_class, "2Dfloat3");
  declare_variable_with_store_creation<2, float4>(m, store_class, "2Dfloat4");
  declare_variable_with_store_creation<4, float4>(m, store_class, "4Dfloat4");

  declare_variable_with_store_creation<1, double>(m, store_class, "1Ddouble");
  declare_variable_with_store_creation<2, double>(m, store_class, "2Ddouble");
  declare_variable_with_store_creation<1, double3>(m, store_class, "1Ddouble3");
  declare_variable_with_store_creation<2, double3>(m, store_class, "2Ddouble3");
  declare_variable_with_store_creation<2, double4>(m, store_class, "2Ddouble4");
  declare_variable_with_store_creation<4, double4>(m, store_class, "4Ddouble4");

  declare_variable_with_store_creation<1, uint>(m, store_class, "1Duint");
  declare_variable_with_store_creation<3, uint>(m, store_class, "3Duint");

  py::enum_<NumericType>(m, "NumericType")
      .value("f32", NumericType::f32)
      .value("f64", NumericType::f64)
      .value("i32", NumericType::i32)
      .value("u32", NumericType::u32)
      .value("undefined", NumericType::undefined);

  declare_vector3<float3, float>(m, "float3");
  declare_vector3<double3, double>(m, "double3");
  declare_vector3<int3, int>(m, "int3");
  declare_vector3<uint3, uint>(m, "uint3");

  declare_vector4<float4, float>(m, "float4");
  declare_vector4<double4, double>(m, "double4");

  declare_pile<float>(m, "float");
  declare_pile<double>(m, "double");

  declare_solver<float>(m, "float");
  declare_solver<double>(m, "double");

  declare_solver_df<float>(m, "float");
  declare_solver_df<double>(m, "double");

  declare_solver_ii<float>(m, "float");
  declare_solver_ii<double>(m, "double");

  declare_display_proxy<float>(m, "float");
  declare_display_proxy<double>(m, "double");

  declare_const<float>(m, "float");
  declare_const<double>(m, "double");

  py::class_<ConstiN>(m, "ConstiN")
      .def_readonly("num_boundaries", &ConstiN::num_boundaries)
      .def_readonly("max_num_contacts", &ConstiN::max_num_contacts)
      .def_readwrite("max_num_particles_per_cell",
                     &ConstiN::max_num_particles_per_cell)
      .def_readwrite("grid_res", &ConstiN::grid_res)
      .def_readwrite("grid_offset", &ConstiN::grid_offset)
      .def_readwrite("max_num_neighbors_per_particle",
                     &ConstiN::max_num_neighbors_per_particle);

  declare_runner<float>(m, "float");
  declare_runner<double>(m, "double");

  py::class_<Mesh>(m, "Mesh")
      .def(py::init<>())
      .def("set_box", &Mesh::set_box)
      .def("set_uv_sphere", &Mesh::set_uv_sphere)
      .def("set_cylinder", &Mesh::set_cylinder)
      .def("set_obj", &Mesh::set_obj)
      .def("calculate_normals", &Mesh::calculate_normals)
      .def("translate", &Mesh::translate)
      .def("clear", &Mesh::clear);

  py::module m_dg = m.def_submodule("dg", "Discregrid");

  declare_distance<float3, float>(m_dg, "float");
  declare_distance<double3, double>(m_dg, "double");

  declare_sphere_distance<float3, float>(m_dg, "float");
  declare_sphere_distance<double3, double>(m_dg, "double");
  declare_box_distance<float3, float>(m_dg, "float");
  declare_box_distance<double3, double>(m_dg, "double");
  declare_cylinder_distance<float3, float>(m_dg, "float");
  declare_cylinder_distance<double3, double>(m_dg, "double");
}
