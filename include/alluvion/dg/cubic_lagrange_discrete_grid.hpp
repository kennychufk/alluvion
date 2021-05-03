#ifndef ALLUVION_DG_CUBIC_LAGRANGE_DISCRETE_GRID_HPP
#define ALLUVION_DG_CUBIC_LAGRANGE_DISCRETE_GRID_HPP

#include "alluvion/dg/discrete_grid.hpp"

namespace alluvion {
namespace dg {

class CubicLagrangeDiscreteGrid : public DiscreteGrid {
 public:
  CubicLagrangeDiscreteGrid(std::string const& filename);
  CubicLagrangeDiscreteGrid(AlignedBox3r const& domain,
                            std::array<unsigned int, 3> const& resolution);

  void save(std::string const& filename) const override;
  void load(std::string const& filename) override;

  unsigned int addFunction(ContinuousFunction const& func, bool verbose = false,
                           SamplePredicate const& pred = nullptr) override;

  std::size_t nCells() const { return m_n_cells; };
  F interpolate(unsigned int field_id, Vector3r const& xi,
                Vector3r* gradient = nullptr) const override;

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
  bool determineShapeFunctions(
      unsigned int field_id, Vector3r const& x,
      std::array<unsigned int, 32>& cell, Vector3r& c0,
      Eigen::Matrix<F, 32, 1>& N,
      Eigen::Matrix<F, 32, 3>* dN = nullptr) const override;

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
  F interpolate(unsigned int field_id, Vector3r const& xi,
                const std::array<unsigned int, 32>& cell, const Vector3r& c0,
                const Eigen::Matrix<F, 32, 1>& N, Vector3r* gradient = nullptr,
                Eigen::Matrix<F, 32, 3>* dN = nullptr) const override;

  void reduceField(unsigned int field_id, Predicate pred) override;

  void forEachCell(unsigned int field_id,
                   std::function<void(unsigned int, AlignedBox3r const&,
                                      unsigned int)> const& cb) const;

  // Data getters
  std::vector<std::vector<F>>& node_data() { return m_nodes; }
  std::vector<std::vector<F>> const& node_data() const { return m_nodes; }
  std::vector<std::vector<std::array<unsigned int, 32>>>& cell_data() {
    return m_cells;
  }
  std::vector<std::vector<std::array<unsigned int, 32>>> const& cell_data()
      const {
    return m_cells;
  }
  std::vector<std::vector<unsigned int>>& cell_map_data() { return m_cell_map; }
  std::vector<std::vector<unsigned int>> const& cell_map_data() const {
    return m_cell_map;
  }
  Vector3r indexToNodePosition(unsigned int l) const;

  std::vector<std::vector<F>> m_nodes;
  std::vector<std::vector<std::array<unsigned int, 32>>> m_cells;
  std::vector<std::vector<unsigned int>> m_cell_map;
};

}  // namespace dg
}  // namespace alluvion
#endif
