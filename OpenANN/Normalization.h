#ifndef OPENANN_NORMALIZATION_H_
#define OPENANN_NORMALIZATION_H_

#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class Normalization
 * Normalize data so that for each feature the mean is 0 and the standard
 * deviation is 1.
 */
class Normalization
{
  Eigen::MatrixXd mean;
  Eigen::MatrixXd std;
public:
  Normalization();

  Normalization& fit(const Eigen::MatrixXd& X);
  Eigen::MatrixXd transform(const Eigen::MatrixXd& X);

  Eigen::VectorXd getMean();
  Eigen::VectorXd getStd();
};

} // OpenANN

#endif // OPENANN_NORMALIZATION_H_
