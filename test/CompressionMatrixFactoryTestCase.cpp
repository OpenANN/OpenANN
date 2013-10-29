#include "CompressionMatrixFactoryTestCase.h"
#include <OpenANN/CompressionMatrixFactory.h>
#include <OpenANN/Compressor.h>
#include <OpenANN/util/Random.h>

void CompressionMatrixFactoryTestCase::run()
{
  RUN(CompressionMatrixFactoryTestCase, compress);
  RUN(CompressionMatrixFactoryTestCase, compressor);
}

void CompressionMatrixFactoryTestCase::setUp()
{
  OpenANN::RandomNumberGenerator().seed(0);
}

void CompressionMatrixFactoryTestCase::compress()
{
  OpenANN::CompressionMatrixFactory cmf;
  cmf.inputDim = 3;
  cmf.paramDim = 3;
  Eigen::MatrixXd compressionMatrix(cmf.paramDim, cmf.inputDim);
  cmf.createCompressionMatrix(compressionMatrix);

  Eigen::MatrixXd expectedOrthogonalFunctionsMatrix(cmf.paramDim, cmf.inputDim);
  expectedOrthogonalFunctionsMatrix <<
                                    1,  1,  1,
                                        1,  0,  -1,
                                        1,  -1,  1;

  for(int m = 0; m < cmf.inputDim; m++)
  {
    for(int i = 0; i < cmf.paramDim; i++)
    {
      ASSERT_EQUALS_DELTA(expectedOrthogonalFunctionsMatrix(m, i), compressionMatrix(m, i), 1e-10);
    }
  }

  ASSERT(fabs(compressionMatrix.determinant()) > 0.1);
}

void CompressionMatrixFactoryTestCase::compressor()
{
  int N = 100;
  int D = 100;
  int F = 10;
  OpenANN::Compressor compressor(D, F, OpenANN::CompressionMatrixFactory::SPARSE_RANDOM);
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd Y = compressor.transform(X);
  ASSERT_EQUALS(Y.rows(), N);
  ASSERT_EQUALS(Y.cols(), F);
  ASSERT_EQUALS_DELTA(X.mean(), 0.0, 0.01);
}
