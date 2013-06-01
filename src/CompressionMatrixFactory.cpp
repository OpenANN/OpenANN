#include <OpenANN/CompressionMatrixFactory.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/util/OpenANNException.h>

namespace OpenANN
{

CompressionMatrixFactory::CompressionMatrixFactory()
  : compress(true), transformation(DCT), inputDim(1), paramDim(1)
{
}

CompressionMatrixFactory::CompressionMatrixFactory(int inputDim,
                                                   int paramDim, Transformation transformation)
  : compress(true), transformation(transformation), inputDim(inputDim),
    paramDim(paramDim)
{
  if(inputDim < 1)
    throw OpenANNException("inputDim has to be at least 1.");
  if(paramDim < 1)
    throw OpenANNException("paramDim has to be at least 1.");
}

void CompressionMatrixFactory::createCompressionMatrix(Eigen::MatrixXd& cm)
{
  if(cm.rows() < paramDim || cm.cols() < inputDim)
    cm.resize(paramDim, inputDim);
  if(compress)
  {
    cm.fill(0.0);
    fillCompressionMatrix(cm);
  }
  else
    cm = Eigen::MatrixXd::Identity(cm.rows(), cm.cols());
}

void CompressionMatrixFactory::fillCompressionMatrix(Eigen::MatrixXd& cm)
{
  RandomNumberGenerator rng;
  const int compressionRatio = cm.cols() / cm.rows();
  for(int i = 0; i < inputDim; i++)
  {
    const double ti = inputDim < 2 ? 0.0 : (double) i / (double)(inputDim - 1);
    for(int m = 0; m < paramDim; m++)
    {
      switch(transformation)
      {
      case DCT:
        cm(m, i) = std::cos((double) m * M_PI * ti);
        break;
      case GAUSSIAN:
        cm(m, i) = rng.sampleNormalDistribution<double>() / (double) paramDim;
        break;
      case SPARSE_RANDOM:
      {
        double r = rng.generate<double>(0.0, 1.0);
        if(r < 1.0 / 6.0)
          cm(m, i) = 1.0 / std::sqrt((double) paramDim);
        else if(r < 5.0 / 6.0)
          cm(m, i) = 0.0;
        else
          cm(m, i) = -1.0 / std::sqrt((double) paramDim);
        break;
      }
      case AVERAGE:
      {
        if(i / compressionRatio == m)
          cm(m, i) = 1.0 / (double) compressionRatio;
      }
      break;
      case EDGE:
      {
        if(i / compressionRatio == m)
        {
          if(i % compressionRatio < compressionRatio / 2)
            cm(m, i) = -2.0 / (double) compressionRatio;
          else
            cm(m, i) = 2.0 / (double) compressionRatio;
        }
      }
      break;
      default:
        cm(m, i) = m == i ? 1.0 : 0.0;
        break;
      }
    }
  }
}

} // namespace OpenANN
