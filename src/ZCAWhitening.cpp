#include <OpenANN/ZCAWhitening.h>
#include <Eigen/Eigenvalues>
#include <cmath>

namespace OpenANN
{

Transformer& ZCAWhitening::fit(const Eigen::MatrixXd& X)
{
  const int N = X.rows();
  const int D = X.cols();
  Eigen::MatrixXd C = X.transpose() * X / ((double) N - 1.0);
  C += Eigen::MatrixXd::Identity(D, D) * 1e-5; // To avoid numerical problems
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenDecomp(C);
  W = eigenDecomp.operatorInverseSqrt();
  return *this;
}

Eigen::MatrixXd ZCAWhitening::transform(const Eigen::MatrixXd& X)
{
  return X * W;
}

}
