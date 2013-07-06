#include "PreprocessingTestCase.h"
#include <OpenANN/Preprocessing.h>
#include <OpenANN/util/OpenANNException.h>

void PreprocessingTestCase::run()
{
  RUN(PreprocessingTestCase, scaling);
  RUN(PreprocessingTestCase, testSampleRandomPatches);
}

void PreprocessingTestCase::scaling()
{
  Eigen::MatrixXd data = Eigen::MatrixXd::Random(20, 20);
  OpenANN::scaleData(data, 5.0, 10.0);
  ASSERT_EQUALS_DELTA(data.minCoeff(), 5.0, 1e-3);
  ASSERT_EQUALS_DELTA(data.maxCoeff(), 10.0, 1e-3);
  OpenANN::scaleData(data, -5.0, 5.0);
  ASSERT_EQUALS_DELTA(data.minCoeff(), -5.0, 1e-3);
  ASSERT_EQUALS_DELTA(data.maxCoeff(), 5.0, 1e-3);
  try
  {
    OpenANN::scaleData(data, 1.0, -1.0);
    ASSERT(false);
  }
  catch(OpenANN::OpenANNException& e)
  {
  }
}

void PreprocessingTestCase::testSampleRandomPatches()
{
  int images = 2;
  int channels = 2;
  int rows = 3;
  int cols = 3;
  Eigen::MatrixXd original(2, channels*rows*cols);
  /* Channel:  #1         #2
   *         1  2  3   10 11 12
   *         4  5  6   13 14 15
   *         7  8  9   16 17 18
   */
  for(int i = 0; i < original.cols(); i++)
  {
    original(0, i) = i+1;
    original(1, i) = i+1;
  }

  int samples = 2;
  int patchRows = 2;
  int patchCols = 2;
  Eigen::MatrixXd patches = OpenANN::sampleRandomPatches(original, channels,
      rows, cols, samples, patchRows, patchCols);
  ASSERT_EQUALS(patches.rows(), images*samples);
  ASSERT_EQUALS(patches.cols(), channels*patchRows*patchCols);
  for(int n = 0; n < patches.rows(); n++)
  {
    ASSERT_EQUALS(patches(n, 0)+1, patches(n, 1));
    ASSERT_EQUALS(patches(n, 1)+2, patches(n, 2));
    ASSERT_EQUALS(patches(n, 2)+1, patches(n, 3));
    ASSERT_EQUALS(patches(n, 4)+1, patches(n, 5));
    ASSERT_EQUALS(patches(n, 5)+2, patches(n, 6));
    ASSERT_EQUALS(patches(n, 6)+1, patches(n, 7));
  }
}
