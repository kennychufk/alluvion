#ifndef ALLUVION_DG_DISCRETE_GRID_HPP
#define ALLUVION_DG_DISCRETE_GRID_HPP

#include <Eigen/Dense>
#include <array>
#include <fstream>
#include <vector>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {

class DiscreteGrid {
 public:
  using CoefficientVector = Eigen::Matrix<F, 32, 1>;
  using ContinuousFunction = std::function<F(Vector3r const&)>;
  using MultiIndex = std::array<unsigned int, 3>;
  using Predicate = std::function<bool(Vector3r const&, F)>;
  using SamplePredicate = std::function<bool(Vector3r const&)>;

  DiscreteGrid() = default;
  DiscreteGrid(AlignedBox3r const& domain,
               std::array<unsigned int, 3> const& resolution)
      : m_domain(domain), m_resolution(resolution), m_n_fields(0u) {
    auto n = Eigen::Matrix<unsigned int, 3, 1>::Map(resolution.data());
    m_cell_size = domain.diagonal().cwiseQuotient(n.cast<F>());
    m_inv_cell_size = m_cell_size.cwiseInverse();
    m_n_cells = n.prod();
  }
  virtual ~DiscreteGrid() = default;

  virtual void save(std::string const& filename) const = 0;
  virtual void load(std::string const& filename) = 0;

  virtual unsigned int addFunction(ContinuousFunction const& func,
                                   bool verbose = false,
                                   SamplePredicate const& pred = nullptr) = 0;

  F interpolate(Vector3r const& xi, Vector3r* gradient = nullptr) const {
    return interpolate(0u, xi, gradient);
  }

  virtual F interpolate(unsigned int field_id, Vector3r const& xi,
                        Vector3r* gradient = nullptr) const = 0;

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
  virtual bool determineShapeFunctions(
      unsigned int field_id, Vector3r const& x,
      std::array<unsigned int, 32>& cell, Vector3r& c0,
      Eigen::Matrix<F, 32, 1>& N,
      Eigen::Matrix<F, 32, 3>* dN = nullptr) const = 0;

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
   * @return F Results of the evaluation of the discrete function at point xi
   */
  virtual F interpolate(unsigned int field_id, Vector3r const& xi,
                        const std::array<unsigned int, 32>& cell,
                        const Vector3r& c0, const Eigen::Matrix<F, 32, 1>& N,
                        Vector3r* gradient = nullptr,
                        Eigen::Matrix<F, 32, 3>* dN = nullptr) const = 0;

  virtual void reduceField(unsigned int field_id, Predicate pred) {}

  MultiIndex singleToMultiIndex(unsigned int i) const;
  unsigned int multiToSingleIndex(MultiIndex const& ijk) const;

  AlignedBox3r subdomain(MultiIndex const& ijk) const;
  AlignedBox3r subdomain(unsigned int l) const;

  AlignedBox3r const& domain() const { return m_domain; }
  std::array<unsigned int, 3> const& resolution() const {
    return m_resolution;
  };
  Vector3r const& cellSize() const { return m_cell_size; }
  Vector3r const& invCellSize() const { return m_inv_cell_size; }

 protected:
  AlignedBox3r m_domain;
  std::array<unsigned int, 3> m_resolution;
  Vector3r m_cell_size;
  Vector3r m_inv_cell_size;
  std::size_t m_n_cells;
  std::size_t m_n_fields;
};
}  // namespace dg
}  // namespace alluvion
#endif
