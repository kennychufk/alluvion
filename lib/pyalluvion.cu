#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "alluvion/constants.hpp"
#include "alluvion/dg/infinite_cylinder_distance.hpp"
#include "alluvion/dg/sphere_distance.hpp"
#include "alluvion/display.hpp"
#include "alluvion/float_shorthands.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/runner.hpp"
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
  py::class_<TPile>(m, class_name.c_str()).def(py::init<Store&, U>());
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
      .def(py::init<F>());
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
          .def("map_graphical_pointers", &Store::map_graphical_pointers)
          .def("unmap_graphical_pointers", &Store::unmap_graphical_pointers);

  declare_variable_with_store_creation<1, float>(m, store_class, "1Dfloat");
  declare_variable_with_store_creation<1, float3>(m, store_class, "1Dfloat3");
  declare_variable_with_store_creation<2, float3>(m, store_class, "2Dfloat3");
  declare_variable_with_store_creation<1, double>(m, store_class, "1Ddouble");
  declare_variable_with_store_creation<1, double3>(m, store_class, "1Ddouble3");
  declare_variable_with_store_creation<2, double3>(m, store_class, "2Ddouble3");

  py::enum_<NumericType>(m, "NumericType")
      .value("f32", NumericType::f32)
      .value("f64", NumericType::f64)
      .value("i32", NumericType::i32)
      .value("u32", NumericType::u32)
      .value("undefined", NumericType::undefined);

  py::class_<Display>(m, "Display");

  declare_pile<float3, float4, float4>(m, "float");
  declare_pile<double3, double4, double4>(m, "double");

  py::module m_dg = m.def_submodule("dg", "Discregrid");

  declare_distance<float3, float>(m_dg, "float");
  declare_distance<double3, double>(m_dg, "double");

  declare_sphere_distance<float3, float>(m_dg, "float");
  declare_sphere_distance<double3, double>(m_dg, "double");
  m.def(
      "evaluate_developing_hagen_poiseuille",
      [](I kQ, std::string particle_x_filename, F pressure_gradient_acc_x,
         F viscosity, F boundary_viscosity, F dt,
         std::vector<F> const& sample_ts) -> std::vector<F> {
        Store store;
        Runner runner;

        F particle_radius = 0.0025_F;
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
        store.get_cn<F>().set_gravity(pressure_gradient_acc);
        store.get_cn<F>().set_boundary_epsilon(1e-9_F);
        F vorticity = 0.01_F;
        F inertia_inverse = 0.1_F;
        F viscosity_omega = 0.5_F;
        F surface_tension_coeff = 0.05_F;
        F surface_tension_boundary_coeff = 0.01_F;
        store.get_cn<F>().set_advanced_fluid_attr(
            viscosity, vorticity, inertia_inverse, viscosity_omega,
            surface_tension_coeff, surface_tension_boundary_coeff);

        I kM = 2;
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
        store.get_cni().set_num_boundaries(pile.get_size());
        store.get_cn<F>().set_contact_tolerance(0.05_F);

        // particles
        U max_num_particles = static_cast<U>(
            2._F * kPi<F> * kQ * kQ * kM * kernel_radius * kernel_radius *
            kernel_radius * density0 / particle_mass);

        // grid
        U3 grid_res{static_cast<U>(kQ * 2), static_cast<U>(kM * 2),
                    static_cast<U>(kQ * 2)};
        I3 grid_offset{-kQ, -kM, -kQ};
        U max_num_particles_per_cell = 64;
        U max_num_neighbors_per_particle = 64;
        store.get_cni().init_grid_constants(grid_res, grid_offset);
        store.get_cni().set_max_num_particles_per_cell(
            max_num_particles_per_cell);
        store.get_cni().set_max_num_neighbors_per_particle(
            max_num_neighbors_per_particle);
        store.get_cn<F>().set_wrap_length(grid_res.x * kernel_radius);

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
        Variable<1, F> particle_pressure =
            store.create<1, F>({max_num_particles});
        Variable<1, F> particle_last_pressure =
            store.create<1, F>({max_num_particles});
        Variable<1, F> particle_aii = store.create<1, F>({max_num_particles});
        Variable<1, F3> particle_dii = store.create<1, F3>({max_num_particles});
        Variable<1, F3> particle_dij_pj =
            store.create<1, F3>({max_num_particles});
        Variable<1, F> particle_sum_tmp =
            store.create<1, F>({max_num_particles});
        Variable<1, F> particle_adv_density =
            store.create<1, F>({max_num_particles});
        Variable<1, F3> particle_pressure_accel =
            store.create<1, F3>({max_num_particles});
        Variable<1, F> particle_density_err =
            store.create<1, F>({max_num_particles});
        Variable<4, Q> pid = store.create<4, Q>(
            {grid_res.x, grid_res.y, grid_res.z, max_num_particles_per_cell});
        Variable<3, U> pid_length =
            store.create<3, U>({grid_res.x, grid_res.y, grid_res.z});
        Variable<2, Q> particle_neighbors = store.create<2, Q>(
            {max_num_particles, max_num_neighbors_per_particle});
        Variable<1, U> particle_num_neighbors =
            store.create<1, U>({max_num_particles});

        SolverIi<F3, Q, F> solver_ii(
            runner, pile, particle_x, particle_normalized_attr, particle_v,
            particle_a, particle_density, particle_boundary_xj,
            particle_boundary_volume, particle_force, particle_torque,
            particle_cfl_v2, particle_pressure, particle_last_pressure,
            particle_aii, particle_dii, particle_dij_pj, particle_sum_tmp,
            particle_adv_density, particle_pressure_accel, particle_density_err,
            pid, pid_length, particle_neighbors, particle_num_neighbors);
        solver_ii.dt = dt;
        solver_ii.max_dt = 0.005;
        solver_ii.min_dt = 0.0001;
        solver_ii.cfl = 0.04;
        solver_ii.particle_radius = particle_radius;

        // sample points
        U num_sample_planes = 14;
        U num_samples_per_plane = 31;
        U num_samples = num_samples_per_plane * num_sample_planes;
        Variable<1, F3> sample_x = store.create<1, F3>({num_samples});
        Variable<1, F3> sample_data3 = store.create<1, F3>({num_samples});
        Variable<2, Q> sample_neighbors =
            store.create<2, Q>({num_samples, max_num_neighbors_per_particle});
        Variable<1, U> sample_num_neighbors = store.create<1, U>({num_samples});
        Variable<2, F3> sample_boundary_xj =
            store.create<2, F3>({pile.get_size(), num_samples});
        Variable<2, F> sample_boundary_volume =
            store.create<2, F>({pile.get_size(), num_samples});
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
        solver_ii.num_particles =
            particle_x.read_file(particle_x_filename.c_str());

        while (true) {
          if (t >= sample_ts[sampling_cursor]) {
            pid_length.set_zero();
            Runner::launch(
                solver_ii.num_particles, 256, [&](U grid_size, U block_size) {
                  update_particle_grid<<<grid_size, block_size>>>(
                      particle_x, pid, pid_length, solver_ii.num_particles);
                });
            Runner::launch(num_samples, 256, [&](U grid_size, U block_size) {
              make_neighbor_list<1><<<grid_size, block_size>>>(
                  sample_x, pid, pid_length, sample_neighbors,
                  sample_num_neighbors, num_samples);
            });
            Runner::launch(
                solver_ii.num_particles, 256, [&](U grid_size, U block_size) {
                  compute_density<<<grid_size, block_size>>>(
                      particle_x, particle_neighbors, particle_num_neighbors,
                      particle_density, particle_boundary_xj,
                      particle_boundary_volume, solver_ii.num_particles);
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
                  sample_data3_host[i].x / num_sample_planes;
            }
            ++sampling_cursor;
            if (sampling_cursor >= sample_ts.size()) {
              return vx;
            }
          }

          solver_ii.step<1>();
          t += solver_ii.dt;
          step_id += 1;
        }

        return vx;
      });
}
