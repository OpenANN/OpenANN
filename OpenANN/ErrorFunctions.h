#ifndef OPENANN_ERROR_FUNCTIONS_H_
#define OPENANN_ERROR_FUNCTIONS_H_

#include <Eigen/Dense>

namespace OpenANN
{

template<typename Derived1, typename Derived2>
double crossEntropy(const Eigen::MatrixBase<Derived1>& Y,
                    const Eigen::MatrixBase<Derived2>& T)
{
  return -(T.array() * ((Y.array() + 1e-10).log())).sum() / (double) Y.rows();
}

template<typename Derived>
double meanSquaredError(const Eigen::MatrixBase<Derived>& YmT)
{
  return YmT.array().square().sum() / (2.0 * (double) YmT.rows());
}

} // namespace OpenANN

#endif // OPENANN_ERROR_FUNCTIONS_H_
