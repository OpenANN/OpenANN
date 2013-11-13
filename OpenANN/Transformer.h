#ifndef OPENANN_TRANSFORMER_H_
#define OPENANN_TRANSFORMER_H_

#include <Eigen/Core>

namespace OpenANN
{

/**
 * @class Transformer
 *
 * Common base for all transformations.
 *
 * Transformations are (usually unsupervised) data preprocessing methods.
 */
class Transformer
{
public:
  virtual ~Transformer() {}
  /**
   * Fit transformation according to training set X.
   * @param X each row represents an instance
   * @return this for chaining
   */
  virtual Transformer& fit(const Eigen::MatrixXd& X) = 0;
  /**
   * Fit transformation according to subset of the training set X.
   * This can be used for online adaption of the transformation.
   * @param X each row represents an instance
   * @return this for chaining
   */
  virtual Transformer& fitPartial(const Eigen::MatrixXd& X)
  {
    return fit(X);
  }
  /**
   * Transform the data.
   * @param X each row represents an instance
   * @return transformed data
   */
  virtual Eigen::MatrixXd transform(const Eigen::MatrixXd& X) = 0;
};

} // namespace OpenANN

#endif // OPENANN_TRANSFORMER_H_
