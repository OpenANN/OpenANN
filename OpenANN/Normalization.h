#ifndef OPENANN_NORMALIZATION_H_
#define OPENANN_NORMALIZATION_H_

#include <OpenANN/Transformer.h>
#include <Eigen/Core>

namespace OpenANN
{

/**
 * @class Normalization
 * Normalize data so that for each feature the mean is 0 and the standard
 * deviation is 1.
 */
class Normalization : public Transformer
{
  Eigen::MatrixXd mean;
  Eigen::MatrixXd std;
public:
  Normalization();

  virtual Transformer& fit(const Eigen::MatrixXd& X);
  virtual Eigen::MatrixXd transform(const Eigen::MatrixXd& X);

  /**
   * Get the mean of the original data.
   * @return mean
   */
  Eigen::VectorXd getMean();
  /**
   * Get the standard deviations of the original data.
   * @return standard deviations
   */
  Eigen::VectorXd getStd();
};

} // namespace OpenANN

#endif // OPENANN_NORMALIZATION_H_
