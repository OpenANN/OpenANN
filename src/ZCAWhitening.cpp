#include <OpenANN/ZCAWhitening.h>
#include <Eigen/Eigenvalues>

namespace OpenANN
{

Transformer& ZCAWhitening::fit(const Eigen::MatrixXd& X)
{
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenDecomp(X.transpose()*X);
  W = eigenDecomp.operatorSqrt();
  return *this;
}

Eigen::MatrixXd ZCAWhitening::transform(const Eigen::MatrixXd& X)
{
  return X * W;
}

}
