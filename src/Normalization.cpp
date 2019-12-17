#include <OpenANN/Normalization.h>
#include <OpenANN/util/AssertionMacros.h>

namespace OpenANN
{

Normalization::Normalization()
{
}

Transformer& Normalization::fit(const Eigen::MatrixXd& X)
{
  mean = X.colwise().mean();

  // Unfortunately there does not seem to be a function for the standard
  // deviation in Eigen
  std.resize(mean.rows(), mean.cols());
  std.setZero();
  for(int n = 0; n < X.rows(); ++n)
  {
    for(int d = 0; d < X.cols(); ++d)
    {
      double tmp = X(n, d) - mean(d);
      std(0, d) += tmp*tmp;
    }
  }
  std /= X.rows();
  std.array() = std.array().sqrt();

  // To avoid division by zero, we do not modify the corresponding features
  for(int d = 0; d < X.cols(); ++d)
    if(std(0, d) == 0.0)
      std(0, d) = 1.0;
  return *this;
}

Eigen::MatrixXd Normalization::transform(const Eigen::MatrixXd& X)
{
  OPENANN_CHECK(mean.cols() > 0);
  OPENANN_CHECK_EQUALS(X.cols(), mean.cols());
  Eigen::MatrixXd normalized(X.rows(), X.cols());
  for(int n = 0; n < X.rows(); ++n)
    normalized.row(n).array() = (X.row(n).array() - mean.array()) *
        std.array().inverse();
  return normalized;
}

Eigen::VectorXd Normalization::getMean()
{
  return mean.transpose();
}

Eigen::VectorXd Normalization::getStd()
{
  return std.transpose();
}

} // namespace OpenANN
