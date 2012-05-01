#include "CompressionMatrixFactoryTestCase.h"
#include <CompressionMatrixFactory.h>

using namespace OpenANN;

void CompressionMatrixFactoryTestCase::run()
{
  RUN(CompressionMatrixFactoryTestCase, compress1D);
  RUN(CompressionMatrixFactoryTestCase, compress2D);
  RUN(CompressionMatrixFactoryTestCase, compress3D);
}

void CompressionMatrixFactoryTestCase::compress1D()
{
  CompressionMatrixFactory cmf;
  cmf.dimension = CompressionMatrixFactory::ONE;
  cmf.firstInputDim = 3;
  cmf.firstParamDim = 3;
  Mt compressionMatrix(cmf.firstParamDim, cmf.firstInputDim);
  cmf.createCompressionMatrix(compressionMatrix);

  Mt expectedOrthogonalFunctionsMatrix(cmf.firstParamDim, cmf.firstInputDim);
  expectedOrthogonalFunctionsMatrix <<
      1,  1,  1,
      1,  0,  -1,
      1,  -1,  1;

  for(int m = 0; m < cmf.firstInputDim; m++)
  {
    for(int i = 0; i < cmf.firstParamDim; i++)
    {
      ASSERT_EQUALS_DELTA(expectedOrthogonalFunctionsMatrix(m, i), compressionMatrix(m, i), (fpt) 1e-10);
    }
  }

  ASSERT(fabs(compressionMatrix.determinant()) > (fpt) 0.1);
}

void CompressionMatrixFactoryTestCase::compress2D()
{
  Mt compressionMatrix(9, 9);
  CompressionMatrixFactory cmf;
  cmf.dimension = CompressionMatrixFactory::TWO;
  cmf.firstInputDim = 3;
  cmf.secondInputDim = 3;
  cmf.firstParamDim = 3;
  cmf.secondParamDim = 3;
  cmf.createCompressionMatrix(compressionMatrix);

  Mt expectedOrthogonalFunctionsMatrix(9, 9);
  expectedOrthogonalFunctionsMatrix <<
      1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  0, -1,  1,  0, -1,  1,  0, -1,
      1, -1,  1,  1, -1,  1,  1, -1,  1,
      1,  1,  1,  0,  0,  0, -1, -1, -1,
      1,  0, -1,  0,  0,  0, -1,  0,  1,
      1, -1,  1,  0,  0,  0, -1,  1, -1,
      1,  1,  1, -1, -1, -1,  1,  1,  1,
      1,  0, -1, -1,  0,  1,  1,  0, -1,
      1, -1,  1, -1,  1, -1,  1, -1,  1;

  for(int m = 0; m < 9; m++)
  {
    for(int i = 0; i < 9; i++)
    {
      ASSERT_EQUALS_DELTA(expectedOrthogonalFunctionsMatrix(m, i), compressionMatrix(m, i), (fpt) 1e-10);
    }
  }

  ASSERT(fabs(compressionMatrix.determinant()) > 0.1);
}
#include <iostream>
void CompressionMatrixFactoryTestCase::compress3D()
{
  CompressionMatrixFactory cmf;
  cmf.dimension = CompressionMatrixFactory::THREE;
  cmf.firstInputDim = 4;
  cmf.secondInputDim = 2;
  cmf.firstParamDim = 2;
  cmf.secondParamDim = 2;
  cmf.thirdParamDim = 2;
  std::vector<std::vector<fpt> > t;
  for(int i = 0; i < cmf.firstInputDim; i++)
  {
    std::vector<fpt> tmp(2);
    tmp[0] = i / 2;
    tmp[1] = i % 2;
    t.push_back(tmp);
  }
  cmf.t = &t;
  Mt compressionMatrix(8, 8);
  cmf.createCompressionMatrix(compressionMatrix);
  ASSERT(fabs(compressionMatrix.determinant()) > (fpt) 0.1);
}
