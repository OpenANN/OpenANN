#include "CompressionMatrixFactoryTestCase.h"
#include <OpenANN/CompressionMatrixFactory.h>

void CompressionMatrixFactoryTestCase::run()
{
  RUN(CompressionMatrixFactoryTestCase, compress);
}

void CompressionMatrixFactoryTestCase::compress()
{
  OpenANN::CompressionMatrixFactory cmf;
  cmf.inputDim = 3;
  cmf.paramDim = 3;
  Mt compressionMatrix(cmf.paramDim, cmf.inputDim);
  cmf.createCompressionMatrix(compressionMatrix);

  Mt expectedOrthogonalFunctionsMatrix(cmf.paramDim, cmf.inputDim);
  expectedOrthogonalFunctionsMatrix <<
      1,  1,  1,
      1,  0,  -1,
      1,  -1,  1;

  for(int m = 0; m < cmf.inputDim; m++)
  {
    for(int i = 0; i < cmf.paramDim; i++)
    {
      ASSERT_EQUALS_DELTA(expectedOrthogonalFunctionsMatrix(m, i), compressionMatrix(m, i), (fpt) 1e-10);
    }
  }

  ASSERT(fabs(compressionMatrix.determinant()) > (fpt) 0.1);
}
