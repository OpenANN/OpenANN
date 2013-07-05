#include <OpenANN/PCA.h>
#include <Eigen/SVD>

namespace OpenANN
{

PCA::PCA(int components, bool whiten)
  : components(components), whiten(whiten)
{
}

PCA& PCA::fit(const Eigen::MatrixXd& X)
{
  const int N = X.rows();
  mean = X.colwise().mean().transpose();
  Eigen::MatrixXd aligned = X;
  aligned.rowwise() -= mean;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd;
  svd.compute(aligned, Eigen::ComputeFullV);
  Eigen::VectorXd S = svd.singularValues();
  W = svd.matrixV();
  if(whiten)
  {
    for(int d = 0; d < X.cols(); ++d)
      W.col(d).array() /= S.array();
    W *= std::sqrt((double) N);
  }
  evr.resize(S.rows());
  evr.array() = S.array().square() / (double) N;
  evr /= evr.sum();
  evr.conservativeResize(components);
  return *this;
}

Eigen::MatrixXd PCA::transform(const Eigen::MatrixXd& X)
{
  Eigen::MatrixXd Y = X;
  Y.rowwise() -= mean;
  return Y * W.topRows(components).transpose();
}

Eigen::VectorXd PCA::explainedVarianceRatio()
{
  return evr;
}

} // namespace OpenANN
