#ifndef OPENANN_ERROR_FUNCTIONS_H_
#define OPENANN_ERROR_FUNCTIONS_H_

#include <Eigen/Dense>

namespace OpenANN
{

/**
 * Compute mean cross entropy.
 * @tparam Derived1 matrix type
 * @tparam Derived2 matrix type
 * @param Y each row contains a prediction
 * @param T each row contains a target
 * @return average cross entropy
 */
template<typename Derived1, typename Derived2>
double crossEntropy(const Eigen::MatrixBase<Derived1>& Y,
                    const Eigen::MatrixBase<Derived2>& T)
{
  return -(T.array() * ((Y.array() + 1e-10).log())).sum() / (double) Y.rows();
}

/**
 * Compute mean squared error.
 * @tparam Derived matrix type
 * @param YmT each row contains the difference of a prediction and a target
 * @return mean squared error
 */
template<typename Derived>
double meanSquaredError(const Eigen::MatrixBase<Derived>& YmT)
{
  return YmT.array().square().sum() / (2.0 * (double) YmT.rows());
}

} // namespace OpenANN

#endif // OPENANN_ERROR_FUNCTIONS_H_
