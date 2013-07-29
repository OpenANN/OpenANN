#ifndef OPENANN_ZCA_WHITENING_H_
#define OPENANN_ZCA_WHITENING_H_

#include <OpenANN/Transformer.h>
#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class ZCAWhitening
 *
 * Zero component analysis whitening transformation.
 *
 * The data will be transformed through \f$ Y = X W \f$ such that the
 * covariance matrix \f$ C = \frac{1}{n-1} Y^T Y \f$ will be \f$ I \f$ and
 * \f$ W = W^T \f$.
 */
class ZCAWhitening : public Transformer
{
  Eigen::MatrixXd W;
public:
  virtual Transformer& fit(const Eigen::MatrixXd& X);
  virtual Eigen::MatrixXd transform(const Eigen::MatrixXd& X);
};

} // namespace OpenANN

#endif // OPENANN_ZCA_WHITENING_H_
