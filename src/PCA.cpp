#include <OpenANN/PCA.h>
#include <OpenANN/util/AssertionMacros.h>
#include <Eigen/SVD>

namespace OpenANN
{

PCA::PCA(int components, bool whiten)
  : components(components), whiten(whiten)
{
}

Transformer& PCA::fit(const Eigen::MatrixXd& X)
{
  const int N = X.rows();
  mean = X.colwise().mean();
  Eigen::MatrixXd aligned = X;
  aligned.rowwise() -= mean.transpose();

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(aligned, Eigen::ComputeFullV);
  Eigen::VectorXd S = svd.singularValues() / std::sqrt((double) N);
  W = svd.matrixV();
  if(whiten)
    for(int d = 0; d < X.cols(); ++d)
      W.col(d).array() /= S.array();

  evr.resize(S.rows());
  evr.array() = S.array().square();
  evr /= evr.sum();
  evr.conservativeResize(components);

  return *this;
}

Eigen::MatrixXd PCA::transform(const Eigen::MatrixXd& X)
{
  OPENANN_CHECK(mean.rows() > 0);
  OPENANN_CHECK_EQUALS(X.cols(), mean.rows());
  Eigen::MatrixXd Y = X;
  Y.rowwise() -= mean.transpose();
  return Y * W.topRows(components).transpose();
}

Eigen::VectorXd PCA::explainedVarianceRatio()
{
  return evr;
}

} // namespace OpenANN
