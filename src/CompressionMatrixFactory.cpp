#include <CompressionMatrixFactory.h>
#include <AssertionMacros.h>
#include <Random.h>

namespace OpenANN
{

CompressionMatrixFactory::CompressionMatrixFactory()
  : compress(true), transformation(DCT),inputDim(1),paramDim(1)
{
}

CompressionMatrixFactory::CompressionMatrixFactory(int firstInputDim,
    int firstParamDim, Transformation transformation)
  : compress(true), transformation(transformation), inputDim(firstInputDim), paramDim(firstParamDim)
{
}

void CompressionMatrixFactory::createCompressionMatrix(Mt& cm)
{
  if(cm.rows() < paramDim || cm.cols() < inputDim)
    cm.resize(paramDim, inputDim);
  if(compress)
  {
    cm.fill(0.0);
    fillCompressionMatrix(cm);
  }
  else
    cm = Mt::Identity(cm.rows(), cm.cols());
}

void CompressionMatrixFactory::fillCompressionMatrix(Mt& cm)
{
  RandomNumberGenerator rng;
  const int compressionRatio = cm.cols() / cm.rows();
  for(int i = 0; i < inputDim; i++)
  {
    const fpt ti = inputDim < 2 ? 0.0 : (fpt) i / (fpt) (inputDim - 1);
    for(int m = 0; m < paramDim; m++)
    {
      switch(transformation)
      {
        case DCT:
          cm(m, i) = std::cos((fpt) m * M_PI * ti);
          break;
        case GAUSSIAN:
          cm(m, i) = rng.sampleNormalDistribution<fpt>() / (fpt) paramDim;
          break;
        case SPARSE_RANDOM:
        {
          fpt r = rng.generate<fpt>((fpt) 0, (fpt) 1);
          if(r < (fpt) (1.0/6.0))
            cm(m, i) = (fpt) 1 / std::sqrt((fpt) paramDim);
          else if(r < (fpt) (5.0/6.0))
            cm(m, i) = (fpt) 0;
          else
            cm(m, i) = (fpt) -1 / std::sqrt((fpt) paramDim);
          break;
        }
        case AVERAGE:
          {
          if(i / compressionRatio == m)
            cm(m, i) = 1.0 / (fpt) compressionRatio;
          }
          break;
        case EDGE:
          {
          if(i / compressionRatio == m)
          {
            if(i % compressionRatio < compressionRatio/2)
              cm(m, i) = -2.0 / (fpt) compressionRatio;
            else
              cm(m, i) = 2.0 / (fpt) compressionRatio;
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

}
