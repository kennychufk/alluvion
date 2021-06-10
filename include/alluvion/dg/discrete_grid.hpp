#ifndef ALLUVION_DG_DISCRETE_GRID_HPP
#define ALLUVION_DG_DISCRETE_GRID_HPP

#include <Eigen/Dense>
#include <array>
#include <fstream>
#include <vector>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {

template <typename TF>
class DiscreteGrid {
 public:
  using CoefficientVector = Eigen::Matrix<TF, 32, 1>;
  using ContinuousFunction = std::function<TF(Vector3r<TF> const&)>;
  using MultiIndex = std::array<unsigned int, 3>;
  using Predicate = std::function<bool(Vector3r<TF> const&, TF)>;
  using SamplePredicate = std::function<bool(Vector3r<TF> const&)>;

  DiscreteGrid() = default;
  DiscreteGrid(AlignedBox3r<TF> const& domain,
               std::array<unsigned int, 3> const& resolution)
      : m_domain(domain), m_resolution(resolution), m_n_fields(0u) {
    auto n = Eigen::Matrix<unsigned int, 3, 1>::Map(resolution.data());
    m_cell_size = domain.diagonal().cwiseQuotient(n.cast<TF>());
    m_inv_cell_size = m_cell_size.cwiseInverse();
    m_n_cells = n.prod();
  }
  ~DiscreteGrid() = default;

  void save(std::string const& filename) const {};
  void load(std::string const& filename){};

  unsigned int addFunction(ContinuousFunction const& func, bool verbose = false,
                           SamplePredicate const& pred = nullptr) {
    return 0;
  };

  TF interpolate(Vector3r<TF> const& xi,
                 Vector3r<TF>* gradient = nullptr) const {
    return interpolate(0u, xi, gradient);
  }

  TF interpolate(unsigned int field_id, Vector3r<TF> const& xi,
                 Vector3r<TF>* gradient = nullptr) const {
    return 0;
  }

  /**
   * @brief Determines the shape functions for the discretization with ID
   * field_id at point xi.
   *
   * @param field_id Discretization ID
   * @param x Location where the shape functions should be determined
   * @param cell cell of x
   * @param c0 vector required for the interpolation
   * @param N	shape functions for the cell of x
   * @param dN (Optional) derivatives of the shape functions, required to
   * compute the gradient
   * @return Success of the function.
   */
  bool determineShapeFunctions(unsigned int field_id, Vector3r<TF> const& x,
                               std::array<unsigned int, 32>& cell,
                               Vector3r<TF>& c0, Eigen::Matrix<TF, 32, 1>& N,
                               Eigen::Matrix<TF, 32, 3>* dN = nullptr) const {
    return false;
  };

  /**
   * @brief Evaluates the given discretization with ID field_id at point xi.
   *
   * @param field_id Discretization ID
   * @param xi Location where the discrete function is evaluated
   * @param cell cell of xi
   * @param c0 vector required for the interpolation
   * @param N	shape functions for the cell of xi
   * @param gradient (Optional) if a pointer to a vector is passed the gradient
   * of the discrete function will be evaluated
   * @param dN (Optional) derivatives of the shape functions, required to
   * compute the gradient
   * @return TF Results of the evaluation of the discrete function at point xi
   */
  TF interpolate(unsigned int field_id, Vector3r<TF> const& xi,
                 const std::array<unsigned int, 32>& cell,
                 const Vector3r<TF>& c0, const Eigen::Matrix<TF, 32, 1>& N,
                 Vector3r<TF>* gradient = nullptr,
                 Eigen::Matrix<TF, 32, 3>* dN = nullptr) const {
    return 0;
  };

  void reduceField(unsigned int field_id, Predicate pred) {}

  MultiIndex singleToMultiIndex(unsigned int l) const {
    auto n01 = m_resolution[0] * m_resolution[1];
    auto k = l / n01;
    auto temp = l % n01;
    auto j = temp / m_resolution[0];
    auto i = temp % m_resolution[0];
    return {{i, j, k}};
  }
  unsigned int multiToSingleIndex(MultiIndex const& ijk) const {
    return m_resolution[1] * m_resolution[0] * ijk[2] +
           m_resolution[0] * ijk[1] + ijk[0];
  }

  AlignedBox3r<TF> subdomain(MultiIndex const& ijk) const {
    auto origin =
        m_domain.min() +
        Eigen::Map<Eigen::Matrix<unsigned int, 3, 1> const>(ijk.data())
            .cast<TF>()
            .cwiseProduct(m_cell_size);
    return {origin, origin + m_cell_size};
  }
  AlignedBox3r<TF> subdomain(unsigned int l) const {
    return subdomain(singleToMultiIndex(l));
  }

  AlignedBox3r<TF> const& domain() const { return m_domain; }
  std::array<unsigned int, 3> const& resolution() const {
    return m_resolution;
  };
  Vector3r<TF> const& cellSize() const { return m_cell_size; }
  Vector3r<TF> const& invCellSize() const { return m_inv_cell_size; }

 protected:
  AlignedBox3r<TF> m_domain;
  std::array<unsigned int, 3> m_resolution;
  Vector3r<TF> m_cell_size;
  Vector3r<TF> m_inv_cell_size;
  std::size_t m_n_cells;
  std::size_t m_n_fields;
};
}  // namespace dg
}  // namespace alluvion
#endif
