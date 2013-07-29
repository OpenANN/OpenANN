#include <OpenANN/ZCAWhitening.h>
#include <Eigen/Eigenvalues>
#include <cmath>

namespace OpenANN
{

Transformer& ZCAWhitening::fit(const Eigen::MatrixXd& X)
{
  const int N = X.rows();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenDecomp(X.transpose()*X);
  W = std::sqrt((double) N - 1.0) * eigenDecomp.operatorInverseSqrt();
  return *this;
}

Eigen::MatrixXd ZCAWhitening::transform(const Eigen::MatrixXd& X)
{
  return X * W;
}

}
