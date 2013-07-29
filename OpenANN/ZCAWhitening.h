#ifndef OPENANN_ZCA_WHITENING_H_
#define OPENANN_ZCA_WHITENING_H_

#include <OpenANN/Transformer.h>
#include <Eigen/Dense>

namespace OpenANN
{

/**
 * @class ZCAWhitening
 * Principal component analysis.
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
