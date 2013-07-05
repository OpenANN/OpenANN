#ifndef OPENANN_PCA_H_
#define OPENANN_PCA_H_

#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class PCA
 * Principal component analysis.
 */
class PCA
{
  int components;
  bool whiten;
  Eigen::VectorXd mean;
  Eigen::MatrixXd W;
  Eigen::VectorXd evr;
public:
  /**
   * Create PCA.
   * @param components number of dimensions after transformation
   * @param whiten outputs should have variance 1
   */
  PCA(int components, bool whiten = true);
  PCA& fit(const Eigen::MatrixXd& X);
  Eigen::MatrixXd transform(const Eigen::MatrixXd& X);

  /**
   * Computes the ratio of explained variance for each transformed feature.
   * @return explaned variance ratio, must be within [0, 1] and sum up to 1
   */
  Eigen::VectorXd explainedVarianceRatio();
};

} // OpenANN

#endif // OPENANN_PCA_H_
