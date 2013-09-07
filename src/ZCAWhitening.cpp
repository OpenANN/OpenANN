#include <OpenANN/ZCAWhitening.h>
#include <OpenANN/util/AssertionMacros.h>
#include <Eigen/SVD>
#include <cmath>

namespace OpenANN
{

Transformer& ZCAWhitening::fit(const Eigen::MatrixXd& X)
{
  const int N = X.rows();
  const int D = X.cols();
  mean = X.colwise().mean();
  Eigen::MatrixXd aligned = X;
  aligned.rowwise() -= mean;
  Eigen::MatrixXd cov = aligned.transpose() * aligned / (double) (N-1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeFullV);
  Eigen::VectorXd S = svd.singularValues();
  W = svd.matrixV();
  for(int d = 0; d < D; ++d)
    W.row(d).array() /= S.array().sqrt();
  W *= svd.matrixV().transpose();

  return *this;
}

Eigen::MatrixXd ZCAWhitening::transform(const Eigen::MatrixXd& X)
{
  OPENANN_CHECK(mean.rows() > 0);
  OPENANN_CHECK_EQUALS(X.cols(), mean.cols());
  Eigen::MatrixXd Y = X;
  Y.rowwise() -= mean;
  return Y * W.transpose();
}

}
