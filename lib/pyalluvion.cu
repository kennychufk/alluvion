#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "alluvion/constants.hpp"
#include "alluvion/dg/box_distance.hpp"
#include "alluvion/dg/capsule_distance.hpp"
#include "alluvion/dg/cylinder_distance.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/mesh_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/display.hpp"
#include "alluvion/display_proxy.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
#include "alluvion/solver_df.hpp"
#include "alluvion/solver_i.hpp"
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
           py::arg("src"), py::arg("num_elements") = -1, py::arg("offset") = 0)
      .def("set_from",
           py::overload_cast<VariableClass const&>(&VariableClass::set_from),
           py::arg("src"))
      .def("set_zero", &VariableClass::set_zero)
      .def("set_same", py::overload_cast<int, U, U>(&VariableClass::set_same),
           py::arg("value"), py::arg("num_elements") = -1,
           py::arg("offset") = 0)
      .def("set_same", py::overload_cast<int>(&VariableClass::set_same),
           py::arg("value"))
      .def("scale", py::overload_cast<M>(&VariableClass::scale))
      .def("scale", py::overload_cast<M, U, U>(&VariableClass::scale),
           py::arg("multiplier"), py::arg("num_elements"),
           py::arg("offset") = 0)
      .def("get_type", &VariableClass::get_type)
      .def("get_num_primitives_per_element",
           &VariableClass::get_num_primitives_per_element)
      .def("get_linear_shape", &VariableClass::get_linear_shape)
      .def("get_num_primitives", &VariableClass::get_num_primitives)
      .def("read_file", &VariableClass::read_file)
      .def("write_file", &VariableClass::write_file)
      .def("get_shape", &VariableClass::get_shape);

  std::string graphical_variable_name = std::string("GraphicalVariable") + name;
  py::class_<GraphicalVariableClass, VariableClass>(
      m, graphical_variable_name.c_str())
      .def(py::init<const GraphicalVariableClass&>())
      .def_readonly("vbo", &GraphicalVariableClass::vbo_);
}

template <unsigned int D, typename M>
void declare_pinned_variable(py::module& m, const char* name) {
  using PinnedVariableClass = PinnedVariable<D, M>;
  std::string class_name = std::string("PinnedVariable") + name;
  py::class_<PinnedVariableClass>(m, class_name.c_str())
      .def("__getitem__",
           [](PinnedVariableClass& variable, U key) { return variable(key); })
      .def("__setitem__", [](PinnedVariableClass& variable, U key, M const& v) {
        variable(key) = v;
      });
}

template <typename TF>
void declare_pile(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  using TPile = Pile<TF>;
  using TRunner = Runner<TF>;
  std::string class_name = std::string("Pile") + name;
  py::class_<TPile>(m, class_name.c_str())
      .def(py::init<Store&, TRunner&, U>())
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
      .def_readonly("thickness_list", &TPile::thickness_list_)
      .def_property_readonly("distance_grids", &TPile::get_distance_grids)
      .def_property_readonly("volume_grids", &TPile::get_volume_grids)
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
      .def_static(
          "read",
          [](const char* filename, U num_rigids, py::array_t<unsigned char> x,
             py::array_t<unsigned char> v, py::array_t<unsigned char> q,
             py::array_t<unsigned char> omega) {
            TPile::read(filename, num_rigids,
                        reinterpret_cast<TF3*>(x.mutable_data()),
                        reinterpret_cast<TF3*>(v.mutable_data()),
                        reinterpret_cast<TQ*>(q.mutable_data()),
                        reinterpret_cast<TF3*>(omega.mutable_data()));
          })
      .def("add", &TPile::add, py::arg("distance"), py::arg("resolution"),
           py::arg("sign") = 1, py::arg("thickness") = 0,
           py::arg("collision_mesh") = Mesh(), py::arg("mass") = 0,
           py::arg("restitution") = 1, py::arg("friction") = 0,
           py::arg("inertia_tensor") = TF3{1, 1, 1},
           py::arg("x") = TF3{0, 0, 0}, py::arg("q") = TQ{0, 0, 0, 1},
           py::arg("display_mesh") = Mesh())
      .def("replace", &TPile::replace, py::arg("i"), py::arg("distance"),
           py::arg("resolution"), py::arg("sign") = 1, py::arg("thickness") = 0,
           py::arg("collision_mesh") = Mesh(), py::arg("mass") = 0,
           py::arg("restitution") = 1, py::arg("friction") = 0,
           py::arg("inertia_tensor") = TF3{1, 1, 1},
           py::arg("x") = TF3{0, 0, 0}, py::arg("q") = TQ{0, 0, 0, 1},
           py::arg("display_mesh") = Mesh())
      .def("build_grids", &TPile::build_grids)
      .def("set_gravity", &TPile::set_gravity)
      .def("reallocate_kinematics_on_device",
           &TPile::reallocate_kinematics_on_device)
      .def("reallocate_kinematics_on_pinned",
           &TPile::reallocate_kinematics_on_pinned)
      .def("write_file", &TPile::write_file)
      .def("read_file", &TPile::read_file, py::arg("filename"),
           py::arg("num_rigids") = -1, py::arg("offset") = 0)
      .def("copy_kinematics_to_device", &TPile::copy_kinematics_to_device)
      .def("integrate_kinematics", &TPile::integrate_kinematics)
      .def("calculate_cfl_v2", &TPile::calculate_cfl_v2)
      .def("find_contacts", py::overload_cast<U, U>(&TPile::find_contacts))
      .def("find_contacts", py::overload_cast<>(&TPile::find_contacts))
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
      .def(py::init<TF3>())
      .def_static(
          "create", [](TF3 widths) { return new TBoxDistance(widths); },
          py::return_value_policy::reference);
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
      .def(py::init<TriangleMesh<TF> const&, bool>(), py::arg("mesh"),
           py::arg("precompute_normals") = true)
      .def_static(
          "create",
          [](TriangleMesh<TF> const& mesh, bool precompute_normals) {
            return new TMeshDistance(mesh, precompute_normals);
          },
          py::return_value_policy::reference, py::arg("mesh"),
          py::arg("precompute_normals") = true);
}

template <typename TF>
void declare_triangle_mesh(py::module& m, const char* name) {
  using TTriangleMesh = dg::TriangleMesh<TF>;
  std::string class_name = std::string("TriangleMesh") + name;
  py::class_<TTriangleMesh>(m, class_name.c_str())
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
      .def_readwrite("vorticity_coeff", &TConst::vorticity_coeff)
      .def_readwrite("inertia_inverse", &TConst::inertia_inverse)
      .def_readwrite("viscosity_omega", &TConst::viscosity_omega)
      .def_readwrite("surface_tension_coeff", &TConst::surface_tension_coeff)
      .def_readwrite("surface_tension_boundary_coeff",
                     &TConst::surface_tension_boundary_coeff)
      .def_readwrite("gravity", &TConst::gravity)
      .def_readwrite("axial_gravity", &TConst::axial_gravity)
      .def_readwrite("radial_gravity", &TConst::radial_gravity)
      .def_readwrite("rshift_base_density", &TConst::rshift_base_density)
      .def_readwrite("rshift_density_factor", &TConst::rshift_density_factor)
      .def_readwrite("boundary_vshift", &TConst::boundary_vshift)
      .def_readwrite("rshift_switch_k", &TConst::rshift_switch_k)
      .def_readwrite("boundary_param0", &TConst::boundary_param0)
      .def_readwrite("boundary_param1", &TConst::boundary_param1)
      .def_readwrite("boundary_param2", &TConst::boundary_param2)
      .def_readwrite("boundary_param3", &TConst::boundary_param3)
      .def_readwrite("boundary_vol_factor", &TConst::boundary_vol_factor)
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
      .def(py::init<TRunner&, TPile&, Store&, U, U3, U, bool, bool, bool>(),
           py::arg("runner"), py::arg("pile"), py::arg("store"),
           py::arg("max_num_particles"), py::arg("grid_res"),
           py::arg("num_ushers") = 0, py::arg("enable_surface_tension") = false,
           py::arg("enable_vorticity") = false, py::arg("graphical") = false)
      .def_readonly("max_num_particles", &TSolver::max_num_particles)
      .def_readonly("particle_max_v2", &TSolver::particle_max_v2)
      .def_readonly("pile_max_v2", &TSolver::pile_max_v2)
      .def_readonly("max_v2", &TSolver::max_v2)
      .def_readonly("cfl_dt", &TSolver::cfl_dt)
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
      .def("normalize",
           py::overload_cast<Variable<1, TF3> const*, Variable<1, TF>*, TF, TF>(
               &TSolver::normalize))
      .def("normalize",
           py::overload_cast<Variable<1, TF> const*, Variable<1, TF>*, TF, TF>(
               &TSolver::normalize))
      .def("reset_solving_var", &TSolver::reset_solving_var)
      .def("reset_t", &TSolver::reset_t)
      .def("emit_single", &TSolver::emit_single, py::arg("x"), py::arg("v"))
      .def("emit_circle", &TSolver::emit_circle, py::arg("center"),
           py::arg("v"), py::arg("radius"), py::arg("num_emission"))
      .def("move_particles_naive", &TSolver::move_particles_naive,
           py::arg("exclusion_min") = TF3{1, 1, 1},
           py::arg("exclusion_max") = TF3{-1, -1, -1})
      .def("dictate_ethier_steinman", &TSolver::dictate_ethier_steinman,
           py::arg("a") = kPi<TF> / 4, py::arg("d") = kPi<TF> / 2,
           py::arg("kinematic_viscosity") = 0,
           py::arg("exclusion_min") = TF3{1, 1, 1},
           py::arg("exclusion_max") = TF3{-1, -1, -1})
      .def("set_mask", &TSolver::set_mask, py::arg("mask"), py::arg("box_min"),
           py::arg("box_max"))
      .def("compute_all_boundaries", &TSolver::compute_all_boundaries)
      .def("sample_all_boundaries", &TSolver::sample_all_boundaries)
      .def("update_particle_neighbors",
           &TSolver::template update_particle_neighbors<0>)
      .def("update_particle_neighbors_wrap1",
           &TSolver::template update_particle_neighbors<1>)
      .def("sample_usher", &TSolver::sample_usher);
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
      .def(py::init<TRunner&, TPile&, Store&, U, U3, U, bool, bool, bool>(),
           py::arg("runner"), py::arg("pile"), py::arg("store"),
           py::arg("max_num_particles"), py::arg("grid_res"),
           py::arg("num_ushers") = 0, py::arg("enable_surface_tension") = false,
           py::arg("enable_vorticity") = false, py::arg("graphical") = false)
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
      .def("step", &TSolverDf::template step<0, 0>)
      .def("step_wrap1", &TSolverDf::template step<1, 0>)
      .def("step_wrap1_gravitation1", &TSolverDf::template step<1, 1>);
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
      .def(py::init<TRunner&, TPile&, Store&, U, U3, U, bool, bool, bool>(),
           py::arg("runner"), py::arg("pile"), py::arg("store"),
           py::arg("max_num_particles"), py::arg("grid_res"),
           py::arg("num_ushers") = 0, py::arg("enable_surface_tension") = false,
           py::arg("enable_vorticity") = false, py::arg("graphical") = false)
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
      .def("step", &TSolverIi::template step<0, 0>)
      .def("step_wrap1", &TSolverIi::template step<1, 0>)
      .def("step_wrap1_gravitation1", &TSolverIi::template step<1, 1>);
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
      .def(py::init<TRunner&, TPile&, Store&, U, U3, U, bool, bool, bool>(),
           py::arg("runner"), py::arg("pile"), py::arg("store"),
           py::arg("max_num_particles"), py::arg("grid_res"),
           py::arg("num_ushers") = 0, py::arg("enable_surface_tension") = false,
           py::arg("enable_vorticity") = false, py::arg("graphical") = false)
      .def_readonly("num_density_solve", &TSolverI::num_density_solve)
      .def_readonly("mean_density_error", &TSolverI::mean_density_error)
      .def_readwrite("density_error_tolerance",
                     &TSolverI::density_error_tolerance)
      .def_readwrite("min_density_solve", &TSolverI::min_density_solve)
      .def_readwrite("max_density_solve", &TSolverI::max_density_solve)
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
      .def("step", &TSolverI::template step<0, 0>)
      .def("step_wrap1", &TSolverI::template step<1, 0>)
      .def("step_wrap1_gravitation1", &TSolverI::template step<1, 1>);
}

template <typename TF>
void declare_usher(py::module& m, const char* name) {
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  using TUsher = Usher<TF>;
  std::string class_name = std::string("Usher") + name;
  py::class_<TUsher>(m, class_name.c_str())
      .def(py::init<Store&, U>(), py::arg("store"), py::arg("num_ushers"))
      .def_readonly("num_ushers", &TUsher::num_ushers)
      .def_property_readonly(
          "drive_x", [](TUsher const& usher) { return usher.drive_x.get(); })
      .def_property_readonly(
          "drive_v", [](TUsher const& usher) { return usher.drive_v.get(); })
      .def_property_readonly(
          "drive_kernel_radius",
          [](TUsher const& usher) { return usher.drive_kernel_radius.get(); })
      .def_property_readonly(
          "drive_strength",
          [](TUsher const& usher) { return usher.drive_strength.get(); })
      .def_property_readonly(
          "sample_x", [](TUsher const& usher) { return usher.sample_x.get(); })
      .def_property_readonly(
          "sample_v", [](TUsher const& usher) { return usher.sample_v.get(); })
      .def_property_readonly(
          "sample_density",
          [](TUsher const& usher) { return usher.sample_density.get(); })
      .def("set",
           [](TUsher& usher, const py::array_t<TF>& x, const py::array_t<TF>& v,
              const py::array_t<TF>& drive_kernel_radius,
              const py::array_t<TF>& drive_strength) {
             usher.set(reinterpret_cast<const TF3*>(x.data()),
                       reinterpret_cast<const TF3*>(v.data()),
                       drive_kernel_radius.data(), drive_strength.data());
           })
      .def("set_sample_x", [](TUsher& usher, const py::array_t<TF>& x) {
        usher.set_sample_x(reinterpret_cast<const TF3*>(x.data()));
      });
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
      .def("add_clear", &TDisplayProxy::add_clear)
      .def("run", &TDisplayProxy::run)
      .def("draw", &TDisplayProxy::draw)
      .def("set_camera", &TDisplayProxy::set_camera)
      .def("set_clip_planes", &TDisplayProxy::set_clip_planes);
}

template <typename TF>
py::class_<Runner<TF>> declare_runner(py::module& m, const char* name) {
  using TRunner = Runner<TF>;
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  std::string class_name = std::string("Runner") + name;
  return py::class_<TRunner>(m, class_name.c_str())
      .def(py::init<>())
      .def_readonly("launch_dict", &TRunner::launch_dict_)
      .def_readonly("launch_stat_dict", &TRunner::launch_stat_dict_)
      .def("summarize", &TRunner::summarize)
      .def("launch_create_fluid_block", &TRunner::launch_create_fluid_block,
           py::arg("particle_x"), py::arg("num_particles"), py::arg("offset"),
           py::arg("mode"), py::arg("box_min"), py::arg("box_max"))
      .def("launch_create_fluid_cylinder_sunflower",
           &TRunner::launch_create_fluid_cylinder_sunflower,
           py::arg("particle_x"), py::arg("num_particles"), py::arg("radius"),
           py::arg("num_particles_per_slice"), py::arg("slice_distance"),
           py::arg("y_min"))
      .def("launch_create_fluid_cylinder",
           &TRunner::launch_create_fluid_cylinder, py::arg("particle_x"),
           py::arg("num_particles"), py::arg("offset"), py::arg("radius"),
           py::arg("y_min"), py::arg("y_max"))
      .def("launch_compute_particle_boundary",
           &TRunner::launch_compute_particle_boundary)
      .def("launch_update_particle_grid", &TRunner::launch_update_particle_grid)
      .def("launch_make_neighbor_list",
           &TRunner::template launch_make_neighbor_list<0>)
      .def("launch_make_neighbor_list_wrap1",
           &TRunner::template launch_make_neighbor_list<1>)
      .def("launch_compute_density", &TRunner::launch_compute_density)
      .def("launch_sample_fluid", &TRunner::template launch_sample_fluid<TF>)
      .def("launch_sample_fluid", &TRunner::template launch_sample_fluid<TF3>)
      .def("launch_sample_density", &TRunner::launch_sample_density)
      .def("launch_copy_kinematics_if_within",
           &TRunner::launch_copy_kinematics_if_within)
      .def("launch_copy_kinematics_if_within_masked",
           &TRunner::launch_copy_kinematics_if_within_masked)
      .def("launch_copy_kinematics_if_between",
           &TRunner::launch_copy_kinematics_if_between)
      .def_static("get_fluid_block_num_particles",
                  &TRunner::get_fluid_block_num_particles, py::arg("mode"),
                  py::arg("box_min"), py::arg("box_max"),
                  py::arg("particle_radius"))
      .def_static("get_fluid_cylinder_num_particles",
                  &TRunner::get_fluid_cylinder_num_particles, py::arg("radius"),
                  py::arg("y_min"), py::arg("y_max"),
                  py::arg("particle_radius"))
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
          .def("copy_cnfloat", &Store::template copy_cn<float>)
          .def("copy_cndouble", &Store::template copy_cn<double>)
          .def("map_graphical_pointers", &Store::map_graphical_pointers)
          .def("unmap_graphical_pointers", &Store::unmap_graphical_pointers);

  py::class_<Runner<float>> runner_float = declare_runner<float>(m, "float");
  py::class_<Runner<double>> runner_double =
      declare_runner<double>(m, "double");

  declare_variable<1, float>(m, store_class, &runner_float, nullptr, "1Dfloat");
  declare_variable<1, float3>(m, store_class, &runner_float, nullptr,
                              "1Dfloat3");
  declare_variable<2, float>(m, store_class, &runner_float, nullptr, "2Dfloat");
  declare_variable<2, float3>(m, store_class, &runner_float, nullptr,
                              "2Dfloat3");
  declare_variable<2, float4>(m, store_class, &runner_float, nullptr,
                              "2Dfloat4");
  declare_variable<4, float4>(m, store_class, &runner_float, nullptr,
                              "4Dfloat4");

  declare_variable<1, double>(m, store_class, nullptr, &runner_double,
                              "1Ddouble");
  declare_variable<2, double>(m, store_class, nullptr, &runner_double,
                              "2Ddouble");
  declare_variable<1, double3>(m, store_class, nullptr, &runner_double,
                               "1Ddouble3");
  declare_variable<2, double3>(m, store_class, nullptr, &runner_double,
                               "2Ddouble3");
  declare_variable<2, double4>(m, store_class, nullptr, &runner_double,
                               "2Ddouble4");
  declare_variable<4, double4>(m, store_class, nullptr, &runner_double,
                               "4Ddouble4");

  declare_variable<1, uint>(m, store_class, &runner_float, &runner_double,
                            "1Duint");
  declare_variable<3, uint>(m, store_class, &runner_float, &runner_double,
                            "3Duint");

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
      .def("scale", &Mesh::scale)
      .def("clear", &Mesh::clear)
      .def("export_obj", &Mesh::export_obj);
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

  declare_usher<float>(m, "float");
  declare_usher<double>(m, "double");

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

  py::module m_dg = m.def_submodule("dg", "Discregrid");

  declare_distance<float3, float>(m_dg, "float");
  declare_distance<double3, double>(m_dg, "double");

  declare_sphere_distance<float3, float>(m_dg, "float");
  declare_sphere_distance<double3, double>(m_dg, "double");
  declare_box_distance<float3, float>(m_dg, "float");
  declare_box_distance<double3, double>(m_dg, "double");
  declare_cylinder_distance<float3, float>(m_dg, "float");
  declare_cylinder_distance<double3, double>(m_dg, "double");
  declare_infinite_cylinder_distance<float3, float>(m_dg, "float");
  declare_infinite_cylinder_distance<double3, double>(m_dg, "double");
  declare_capsule_distance<float3, float>(m_dg, "float");
  declare_capsule_distance<double3, double>(m_dg, "double");
  declare_mesh_distance<float3, float>(m_dg, "float");
  declare_mesh_distance<double3, double>(m_dg, "double");
  declare_triangle_mesh<float>(m_dg, "float");
  declare_triangle_mesh<double>(m_dg, "double");
}
