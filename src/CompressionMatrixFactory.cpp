#include <CompressionMatrixFactory.h>
#include <AssertionMacros.h>
#include <Random.h>

namespace OpenANN
{

CompressionMatrixFactory::CompressionMatrixFactory()
  : compress(true), dimension(ONE), transformation(DCT),
    firstInputDim(1), secondInputDim(1),
    firstParamDim(1), secondParamDim(1), thirdParamDim(1),
    t(0)
{
}

CompressionMatrixFactory::CompressionMatrixFactory(int firstInputDim, int firstParamDim,
                                                   Transformation transformation)
  : compress(true), dimension(ONE), transformation(transformation),
    firstInputDim(firstInputDim), secondInputDim(1),
    firstParamDim(firstParamDim), secondParamDim(1), thirdParamDim(1),
    t(0)
{
}

CompressionMatrixFactory::CompressionMatrixFactory(int firstInputDim, int secondInputDim,
                                                   int firstParamDim, int secondParamDim,
                                                   Transformation transformation)
  : compress(true), dimension(TWO), transformation(transformation),
    firstInputDim(firstInputDim), secondInputDim(secondInputDim),
    firstParamDim(firstParamDim), secondParamDim(secondParamDim), thirdParamDim(1),
    t(0)
{
}

CompressionMatrixFactory::CompressionMatrixFactory(int firstInputDim, int secondInputDim, std::vector<std::vector<fpt> >* t,
                                                   int firstParamDim, int secondParamDim, int thirdParamDim,
                                                   Transformation transformation)
  : compress(true), dimension(THREE), transformation(transformation),
    firstInputDim(firstInputDim), secondInputDim(secondInputDim),
    firstParamDim(firstParamDim), secondParamDim(secondParamDim), thirdParamDim(thirdParamDim),
    t(t)
{
}

void CompressionMatrixFactory::createCompressionMatrix(Mt& orthogonalFunctionsMatrix)
{
  if(orthogonalFunctionsMatrix.rows() < firstParamDim * secondParamDim * thirdParamDim ||
    orthogonalFunctionsMatrix.cols() < firstInputDim * secondInputDim)
  {
    orthogonalFunctionsMatrix = Mt::Identity(firstParamDim * secondParamDim * thirdParamDim,
                                     firstInputDim * secondInputDim);
  }
  OPENANN_CHECK(orthogonalFunctionsMatrix.rows() >= firstParamDim * secondParamDim * thirdParamDim);
  OPENANN_CHECK(orthogonalFunctionsMatrix.cols() >= firstInputDim * secondInputDim);
  if(compress)
  {
    orthogonalFunctionsMatrix.fill(0.0);
    switch(dimension)
    {
      case ONE:
        create1DCompressionMatrix(orthogonalFunctionsMatrix);
        break;
      case TWO:
        create2DCompressionMatrix(orthogonalFunctionsMatrix);
        break;
      case THREE:
        create3DCompressionMatrix(orthogonalFunctionsMatrix);
        break;
      default:
        OPENANN_CHECK(false && "Unknown compression dimension.");
    }
  }
  else
  {
    orthogonalFunctionsMatrix = Mt::Identity(orthogonalFunctionsMatrix.rows(),
                                                  orthogonalFunctionsMatrix.cols());
  }
}

void CompressionMatrixFactory::create1DCompressionMatrix(Mt& orthogonalFunctionsMatrix)
{
  RandomNumberGenerator rng;
  const int compressionRatio = orthogonalFunctionsMatrix.cols() / orthogonalFunctionsMatrix.rows();
  for(int i = 0; i < firstInputDim; i++)
  {
    const fpt ti = firstInputDim < 2 ? 0.0 : (fpt) i / (fpt) (firstInputDim - 1);
    for(int m = 0; m < firstParamDim; m++)
    {
      switch(transformation)
      {
        case DCT:
          orthogonalFunctionsMatrix(m, i) = std::cos((fpt) m * M_PI * ti);
          break;
        case GAUSSIAN:
          orthogonalFunctionsMatrix(m, i) = rng.sampleNormalDistribution<fpt>() / (fpt) firstParamDim;
          break;
        case AVERAGE:
          {
          if(i / compressionRatio == m)
            orthogonalFunctionsMatrix(m, i) = 1.0 / (fpt) compressionRatio;
          }
          break;
        case EDGE:
          {
          if(i / compressionRatio == m)
          {
            if(i % compressionRatio < compressionRatio/2)
              orthogonalFunctionsMatrix(m, i) = -2.0 / (fpt) compressionRatio;
            else
              orthogonalFunctionsMatrix(m, i) = 2.0 / (fpt) compressionRatio;
          }
          }
          break;
        default:
          orthogonalFunctionsMatrix(m, i) = m == i ? 1.0 : 0.0;
          break;
      }
    }
  }
}

void CompressionMatrixFactory::create2DCompressionMatrix(Mt& orthogonalFunctionsMatrix)
{
  RandomNumberGenerator rng;
  for(int h = 0; h < firstInputDim; h++)
  {
    for(int l = 0; l < secondInputDim; l++)
    {
      const fpt th = firstInputDim < 2 ? 0.0 : (fpt) h / (fpt) (firstInputDim - 1);
      const fpt tl = secondInputDim < 2 ? 0.0 : (fpt) l / (fpt) (secondInputDim - 1);
      for(int p = 0; p < firstParamDim; p++)
      {
        for(int q = 0; q < secondParamDim; q++)
        {
          const int m = p*secondParamDim+q;
          const int i = h*secondInputDim+l;
          switch(transformation)
          {
            case DCT:
              orthogonalFunctionsMatrix(m, i) = std::cos((fpt) p * M_PI * th) * std::cos((fpt) q * M_PI * tl);
              break;
            case GAUSSIAN:
              orthogonalFunctionsMatrix(m, i) = rng.sampleNormalDistribution<fpt>() * 0.01;
              break;
            default:
              orthogonalFunctionsMatrix(m, i) = m == i ? 1.0 : 0.0;
              break;
          }
        }
      }
    }
  }
}

void CompressionMatrixFactory::create3DCompressionMatrix(Mt& orthogonalFunctionsMatrix)
{
  OPENANN_CHECK(t);
  OPENANN_CHECK_EQUALS(t->size(), firstInputDim);
  for(int h = 0; h < firstInputDim; h++)
  {
    for(int l = 0; l < secondInputDim; l++)
    {
      const fpt tl = secondInputDim < 2 ? 0.0 : (fpt) l / (fpt) (secondInputDim - 1);
      for(int p = 0; p < firstParamDim; p++)
      {
        for(int q = 0; q < secondParamDim; q++)
        {
          for(int r = 0; r < thirdParamDim; r++)
          {
            const int m = p*secondParamDim*thirdParamDim+q*thirdParamDim+r;
            const int i = h*secondInputDim+l;
            switch(transformation)
            {
              case DCT:
                orthogonalFunctionsMatrix(m, i) = std::cos((fpt) p * M_PI * (*t)[h][0])
                    * std::cos((fpt) q * M_PI * (*t)[h][1]) * std::cos((fpt) r * M_PI * tl);
                break;
              default:
                orthogonalFunctionsMatrix(m, i) = m == i ? 1.0 : 0.0;
                break;
            }
          }
        }
      }
    }
  }
}

}
