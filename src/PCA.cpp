#include <OpenANN/PCA.h>
#include <Eigen/SVD>

namespace OpenANN
{

PCA::PCA(bool whiten)
  : whiten(whiten)
{
}

PCA& PCA::fit(const Eigen::MatrixXd& X)
{
  const int N = X.rows();
  mean = X.colwise().mean().transpose();
  Eigen::MatrixXd aligned = X;
  aligned.rowwise() -= mean;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd;
  svd.compute(aligned, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd S = svd.singularValues();
  W = svd.matrixV();
  if(whiten)
  {
    for(int d = 0; d < X.cols(); ++d)
      W.col(d).array() /= S.array();
    W *= std::sqrt((double) N);
  }
}

Eigen::MatrixXd PCA::transform(const Eigen::MatrixXd& X)
{
  Eigen::MatrixXd Y = X;
  Y.rowwise() -= mean;
  return Y * W.transpose();
}

} // namespace OpenANN
