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
#include <iostream>
void PreprocessingTestCase::testSampleRandomPatches()
{
  int channels = 2;
  int rows = 3;
  int cols = 3;
  Eigen::MatrixXd images(1, channels*rows*cols);
  for(int i = 0; i < images.cols(); i++)
    images(0, i) = i+1;

  int samples = 4;
  int patchRows = 2;
  int patchCols = 2;
  Eigen::MatrixXd patches = OpenANN::sampleRandomPatches(images, channels,
      rows, cols, samples, patchRows, patchCols);
  ASSERT_EQUALS(patches.rows(), samples);
  ASSERT_EQUALS(patches.cols(), channels*patchRows*patchCols);
  std::cout << images << std::endl << std::endl;
  std::cout << patches << std::endl << std::endl;
}
