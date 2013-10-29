#include <OpenANN/Compressor.h>

namespace OpenANN
{

Compressor::Compressor(int inputDim, int outputDim,
                       CompressionMatrixFactory::Transformation transformation)
  : cm(outputDim, inputDim)
{
  CompressionMatrixFactory cmf(inputDim, outputDim, transformation);
  cmf.createCompressionMatrix(cm);
}

Transformer& Compressor::fit(const Eigen::MatrixXd& X)
{
  return *this;
}

Transformer& Compressor::fitPartial(const Eigen::MatrixXd& X)
{
  return *this;
}

Eigen::MatrixXd Compressor::transform(const Eigen::MatrixXd& X)
{
  return X * cm.transpose();
}

int Compressor::getOutputs()
{
  return cm.rows();
}

} // namespace OpenANN
