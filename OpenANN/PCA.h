#ifndef OPENANN_PCA_H_
#define OPENANN_PCA_H_

#include <OpenANN/Transformer.h>
#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class PCA
 * Principal component analysis.
 */
class PCA : public Transformer
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

  virtual Transformer& fit(const Eigen::MatrixXd& X);
  virtual Eigen::MatrixXd transform(const Eigen::MatrixXd& X);

  /**
   * Get the ratio of explained variance for each transformed feature.
   * @return explaned variance ratio, must be within [0, 1] and sum up to 1
   *         for all features (including discarded features)
   */
  Eigen::VectorXd explainedVarianceRatio();
};

} // namespace OpenANN

#endif // OPENANN_PCA_H_
