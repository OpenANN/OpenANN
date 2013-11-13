#ifndef OPENANN_COMPRESSOR_H_
#define OPENANN_COMPRESSOR_H_

#include <OpenANN/Transformer.h>
#include <OpenANN/CompressionMatrixFactory.h>
#include <Eigen/Core>

namespace OpenANN
{

/**
 * @class Compressor
 *
 * Compresses arbitrary one-dimensional data.
 *
 * The compression must have the form \f$ \Phi x = y \f$, where \f$ \Phi \f$
 * can be an arbitrary compression matrix, \f$ x \f$ is the D-dimensional
 * input and \f$ y \f$ is the F-dimensional output (i.e. F < D).
 */
class Compressor : public Transformer
{
  Eigen::MatrixXd cm;
public:
  Compressor(int inputDim, int outputDim,
             CompressionMatrixFactory::Transformation transformation);
  virtual Transformer& fit(const Eigen::MatrixXd& X);
  virtual Transformer& fitPartial(const Eigen::MatrixXd& X);
  virtual Eigen::MatrixXd transform(const Eigen::MatrixXd& X);
  int getOutputs();
};

} // namespace OpenANN

#endif // OPENANN_COMPRESSOR_H_
