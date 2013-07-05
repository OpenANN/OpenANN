#ifndef OPENANN_PCA_H_
#define OPENANN_PCA_H_

#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class PCA
 * Principal component analysis.
 *
 * After transformation, the features are ordered by importance from left to
 * right, i.e. some of the right columns can be discarded.
 */
class PCA
{
  bool whiten;
  Eigen::VectorXd mean;
  Eigen::MatrixXd W;
public:
  /**
   * Create PCA.
   * @param whiten output should have variance 1
   */
  PCA(bool whiten = true);
  PCA& fit(const Eigen::MatrixXd& X);
  Eigen::MatrixXd transform(const Eigen::MatrixXd& X);
};

} // OpenANN

#endif // OPENANN_PCA_H_
