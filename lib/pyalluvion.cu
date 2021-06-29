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
#include "alluvion/float_shorthands.hpp"
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
  store_class.def(create_func_name.c_str(), &Store::create<D, M>);
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

template <typename TF3, typename TQ, typename TF>
void declare_pile(py::module& m, const char* name) {
  using TPile = Pile<TF3, TQ, TF>;
  std::string class_name = std::string("Pile") + name;
  py::class_<TPile>(m, class_name.c_str())
      .def(py::init<Store&, U>())
      .def_readonly("distance_grids", &TPile::distance_grids_)
      .def_readonly("volume_grids", &TPile::volume_grids_)
      .def("add", &TPile::add)
      .def("replace", &TPile::replace)
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

template <typename TF3, typename TQ, typename TF>
void declare_solver_df(py::module& m, const char* name) {
  using TSolver = Solver<TF>;
  using TSolverDf = SolverDf<TF3, TQ, TF>;
  using TPile = Pile<TF3, TQ, TF>;
  std::string class_name = std::string("SolverDf") + name;
  py::class_<TSolverDf, TSolver>(m, class_name.c_str())
      .def(py::init<Runner&, TPile&, Variable<1, TF3>&, Variable<1, TF>&,
                    Variable<1, TF3>&, Variable<1, TF3>&, Variable<1, TF>&,
                    Variable<2, TF3>&, Variable<2, TF>&, Variable<2, TF3>&,
                    Variable<2, TF3>&, Variable<1, TF>&, Variable<1, TF>&,
                    Variable<1, TF>&, Variable<1, TF>&, Variable<1, TF>&,
                    Variable<4, TQ>&, Variable<3, U>&, Variable<2, TQ>&,
                    Variable<1, U>&>())
      .def("step", &TSolverDf::template step<0, 0>);
}

template <typename TF3, typename TQ, typename TF>
void declare_solver_ii(py::module& m, const char* name) {
  using TSolver = Solver<TF>;
  using TSolverIi = SolverIi<TF3, TQ, TF>;
  using TPile = Pile<TF3, TQ, TF>;
  std::string class_name = std::string("SolverIi") + name;
  py::class_<TSolverIi, TSolver>(m, class_name.c_str())
      .def(py::init<Runner&, TPile&, Variable<1, TF3>&, Variable<1, TF>&,
                    Variable<1, TF3>&, Variable<1, TF3>&, Variable<1, TF>&,
                    Variable<2, TF3>&, Variable<2, TF>&, Variable<2, TF3>&,
                    Variable<2, TF3>&, Variable<1, TF>&, Variable<1, TF>&,
                    Variable<1, TF>&, Variable<1, TF>&, Variable<1, TF3>&,
                    Variable<1, TF3>&, Variable<1, TF>&, Variable<1, TF>&,
                    Variable<1, TF3>&, Variable<1, TF>&, Variable<4, TQ>&,
                    Variable<3, U>&, Variable<2, TQ>&, Variable<1, U>&>())
      .def("step", &TSolverIi::template step<0>);
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
          .def("get_display_proxy",
               [](Store& store) { return DisplayProxy(store.get_display()); })
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

  py::class_<DisplayProxy>(m, "DisplayProxy")
      .def("create_colormap_viridis", &DisplayProxy::create_colormap_viridis)
      .def("add_particle_shading_programfloat",
           &DisplayProxy::template add_particle_shading_program<float>)
      .def("add_pile_shading_programfloat",
           &DisplayProxy::template add_pile_shading_program<float3, float4,
                                                            float>)
      .def("add_solver_df_step",
           [](DisplayProxy& display_proxy, SolverDf<F3, Q, F>& solver_df,
              Store& store) {
             display_proxy.display_->add_shading_program(new ShadingProgram(
                 nullptr, nullptr, {}, {},
                 [&](ShadingProgram& program, Display& display) {
                   store.map_graphical_pointers();
                   // start of simulation loop
                   for (U frame_interstep = 0; frame_interstep < 10;
                        ++frame_interstep) {
                     solver_df.step<0, 0>();
                   }
                   solver_df.colorize_kappa_v(-0.002_F, 0.0_F);
                   // solver_df.colorize_speed(0, 2);
                   store.unmap_graphical_pointers();
                 }));
           })
      .def("run", &DisplayProxy::run)
      .def("set_camera", &DisplayProxy::set_camera);

  declare_pile<float3, float4, float>(m, "float");
  declare_pile<double3, double4, double>(m, "double");

  declare_solver<float>(m, "float");
  declare_solver<double>(m, "double");

  declare_solver_df<float3, float4, float>(m, "float");
  declare_solver_df<double3, double4, double>(m, "double");

  declare_solver_ii<float3, float4, float>(m, "float");
  declare_solver_ii<double3, double4, double>(m, "double");

  declare_const<float>(m, "float");
  py::class_<ConstiN>(m, "ConstiN")
      .def_readonly("num_boundaries", &ConstiN::num_boundaries)
      .def_readonly("max_num_contacts", &ConstiN::max_num_contacts)
      .def_readwrite("max_num_particles_per_cell",
                     &ConstiN::max_num_particles_per_cell)
      .def_readwrite("grid_res", &ConstiN::grid_res)
      .def_readwrite("grid_offset", &ConstiN::grid_offset)
      .def_readwrite("max_num_neighbors_per_particle",
                     &ConstiN::max_num_neighbors_per_particle);

  py::class_<Runner>(m, "Runner")
      .def(py::init<>())
      .def("create_fluid_block",
           [](Runner& runner, U block_size, Variable<1, float3>& particle_x,
              U num_particles, U offset, int mode, float3 box_min,
              float3 box_max) {
             runner.launch(
                 num_particles, block_size,
                 [&](U grid_size, U block_size) {
                   create_fluid_block<F3, F><<<grid_size, block_size>>>(
                       particle_x, num_particles, offset, mode, box_min,
                       box_max);
                 },
                 "create_fluid_block");
           })
      .def("create_fluid_cylinder",
           [](Runner& runner, U block_size, Variable<1, float3>& particle_x,
              U num_particles, float radius, U num_particles_per_slice,
              float slice_distance, float y_min) {
             runner.launch(
                 num_particles, block_size,
                 [&](U grid_size, U block_size) {
                   create_fluid_cylinder<F3, F><<<grid_size, block_size>>>(
                       particle_x, num_particles, radius,
                       num_particles_per_slice, slice_distance, y_min);
                 },
                 "create_fluid_cylinder");
           });

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

  m.def(
      "evaluate_developing_hagen_poiseuille",
      [](I kQ, std::string particle_x_filename, F pressure_gradient_acc_x,
         F viscosity, F boundary_viscosity, F dt,
         std::vector<F> const& sample_ts) -> std::vector<F> {
        Store store;
        Runner runner;

        F particle_radius = 0.00125_F;
        F kernel_radius = particle_radius * 4.0_F;
        F density0 = 1000.0_F;
        F cubical_particle_volume =
            8 * particle_radius * particle_radius * particle_radius;
        F volume_relative_to_cube = 0.8_F;
        F particle_mass =
            cubical_particle_volume * volume_relative_to_cube * density0;

        F3 pressure_gradient_acc = F3{0._F, pressure_gradient_acc_x, 0._F};

        store.get_cn<F>().set_cubic_discretization_constants();
        store.get_cn<F>().set_kernel_radius(kernel_radius);
        store.get_cn<F>().set_particle_attr(particle_radius, particle_mass,
                                            density0);
        store.get_cn<F>().gravity = pressure_gradient_acc;
        store.get_cn<F>().boundary_epsilon = 1e-9_F;
        store.get_cn<F>().viscosity = viscosity;

        I kM = 4;
        F cylinder_length = 2._F * kM * kernel_radius;
        F R = kernel_radius * kQ;

        // rigids
        F restitution = 1._F;
        F friction = 0._F;
        U max_num_contacts = 512;
        Pile<F3, Q, F> pile(store, max_num_contacts);
        pile.add(new InfiniteCylinderDistance<F3, F>(R), U3{64, 1, 64}, -1._F,
                 0, Mesh(), 0._F, restitution, friction, boundary_viscosity,
                 F3{1, 1, 1}, F3{0, 0, 0}, Q{0, 0, 0, 1}, Mesh());
        pile.build_grids(4 * kernel_radius);
        pile.reallocate_kinematics_on_device();
        store.get_cni().num_boundaries = pile.get_size();
        store.get_cn<F>().contact_tolerance = particle_radius;

        // particles
        U max_num_particles = static_cast<U>(kPi<F> * R * R * cylinder_length *
                                             density0 / particle_mass);

        // grid
        U3 grid_res{static_cast<U>(kQ * 2), static_cast<U>(kM * 2),
                    static_cast<U>(kQ * 2)};
        I3 grid_offset{-kQ, -kM, -kQ};
        U max_num_particles_per_cell = 64;
        U max_num_neighbors_per_particle = 64;
        store.get_cni().grid_res = grid_res;
        store.get_cni().grid_offset = grid_offset;
        store.get_cni().max_num_particles_per_cell = max_num_particles_per_cell;
        store.get_cni().max_num_neighbors_per_particle =
            max_num_neighbors_per_particle;
        store.get_cn<F>().set_wrap_length(grid_res.y * kernel_radius);

        Variable<1, F3> particle_x(store.create<1, F3>({max_num_particles}));
        Variable<1, F> particle_normalized_attr(
            store.create<1, F>({max_num_particles}));
        Variable<1, F3> particle_v = store.create<1, F3>({max_num_particles});
        Variable<1, F3> particle_a = store.create<1, F3>({max_num_particles});
        Variable<1, F> particle_density =
            store.create<1, F>({max_num_particles});
        Variable<2, F3> particle_boundary_xj =
            store.create<2, F3>({pile.get_size(), max_num_particles});
        Variable<2, F> particle_boundary_volume =
            store.create<2, F>({pile.get_size(), max_num_particles});
        Variable<2, F3> particle_force =
            store.create<2, F3>({pile.get_size(), max_num_particles});
        Variable<2, F3> particle_torque =
            store.create<2, F3>({pile.get_size(), max_num_particles});
        Variable<1, F> particle_cfl_v2 =
            store.create<1, F>({max_num_particles});
        Variable<1, F> particle_dfsph_factor =
            store.create<1, F>({max_num_particles});
        Variable<1, F> particle_kappa = store.create<1, F>({max_num_particles});
        Variable<1, F> particle_kappa_v =
            store.create<1, F>({max_num_particles});
        Variable<1, F> particle_density_adv =
            store.create<1, F>({max_num_particles});

        Variable<4, Q> pid = store.create<4, Q>(
            {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell});
        Variable<3, U> pid_length =
            store.create<3, U>({grid_res.x, grid_res.y, grid_res.z});
        Variable<2, Q> particle_neighbors = store.create<2, Q>(
            {max_num_particles, max_num_neighbors_per_particle});
        Variable<1, U> particle_num_neighbors =
            store.create<1, U>({max_num_particles});

        SolverDf<F3, Q, F> solver_df(
            runner, pile, particle_x, particle_normalized_attr, particle_v,
            particle_a, particle_density, particle_boundary_xj,
            particle_boundary_volume, particle_force, particle_torque,
            particle_cfl_v2, particle_dfsph_factor, particle_kappa,
            particle_kappa_v, particle_density_adv, pid, pid_length,
            particle_neighbors, particle_num_neighbors);
        solver_df.dt = dt;
        solver_df.max_dt = 0.005;
        solver_df.min_dt = 0;
        solver_df.cfl = 0.04;
        solver_df.particle_radius = particle_radius;

        // sample points
        U num_sample_planes = 14;
        U num_samples_per_plane = 31;
        U num_samples = num_samples_per_plane * num_sample_planes;
        Variable<1, F3> sample_x = store.create<1, F3>({num_samples});
        Variable<1, F3> sample_data3 = store.create<1, F3>({num_samples});
        Variable<2, Q> sample_neighbors =
            store.create<2, Q>({num_samples, max_num_neighbors_per_particle});
        Variable<1, U> sample_num_neighbors = store.create<1, U>({num_samples});
        {
          std::vector<F3> sample_x_host(num_samples);
          F distance_between_sample_planes =
              cylinder_length / num_sample_planes;
          for (I i = 0; i < num_samples; ++i) {
            I plane_id = i / num_samples_per_plane;
            I id_in_plane = i % num_samples_per_plane;
            sample_x_host[i] = F3{
                R * 2._F / (num_samples_per_plane + 1) *
                    (id_in_plane - static_cast<I>(num_samples_per_plane) / 2),
                cylinder_length * -0.5_F +
                    distance_between_sample_planes * plane_id,
                0._F};
          }
          sample_x.set_bytes(sample_x_host.data());
        }
        std::vector<F3> sample_data3_host(num_samples);
        std::vector<F> vx(num_samples_per_plane * sample_ts.size());

        U step_id = 0;
        F t = 0;

        I sampling_cursor = 0;

        store.copy_cn<F>();
        solver_df.num_particles =
            particle_x.read_file(particle_x_filename.c_str());

        while (true) {
          if (t >= sample_ts[sampling_cursor]) {
            pid_length.set_zero();
            Runner::launch(
                solver_df.num_particles, 256, [&](U grid_size, U block_size) {
                  update_particle_grid<<<grid_size, block_size>>>(
                      particle_x, pid, pid_length, solver_df.num_particles);
                });
            Runner::launch(num_samples, 256, [&](U grid_size, U block_size) {
              make_neighbor_list<1><<<grid_size, block_size>>>(
                  sample_x, pid, pid_length, sample_neighbors,
                  sample_num_neighbors, num_samples);
            });
            Runner::launch(
                solver_df.num_particles, 256, [&](U grid_size, U block_size) {
                  compute_density<<<grid_size, block_size>>>(
                      particle_x, particle_neighbors, particle_num_neighbors,
                      particle_density, particle_boundary_xj,
                      particle_boundary_volume, solver_df.num_particles);
                });
            Runner::launch(num_samples, 256, [&](U grid_size, U block_size) {
              sample_fluid<<<grid_size, block_size>>>(
                  sample_x, particle_x, particle_density, particle_v,
                  sample_neighbors, sample_num_neighbors, sample_data3,
                  num_samples);
            });
            sample_data3.get_bytes(sample_data3_host.data());
            for (I i = 0; i < sample_data3_host.size(); ++i) {
              vx[num_samples_per_plane * sampling_cursor +
                 (i % num_samples_per_plane)] +=
                  sample_data3_host[i].y / num_sample_planes;
            }
            ++sampling_cursor;
            if (sampling_cursor >= sample_ts.size()) {
              return vx;
            }
          }

          solver_df.step<1, 0>();
          t += solver_df.dt;
          step_id += 1;
        }

        return vx;
      });
}
