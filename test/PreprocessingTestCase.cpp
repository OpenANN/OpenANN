#include "PreprocessingTestCase.h"
#include <OpenANN/Preprocessing.h>
#include <OpenANN/util/OpenANNException.h>

void PreprocessingTestCase::run()
{
  RUN(PreprocessingTestCase, scaling);
}

void PreprocessingTestCase::scaling()
{
  Eigen::MatrixXd data = Eigen::MatrixXd::Random(20, 20);
  OpenANN::scaleData(data, (double) 5.0, (double) 10.0);
  ASSERT_EQUALS_DELTA(data.minCoeff(), (double) 5.0, (double) 1e-3);
  ASSERT_EQUALS_DELTA(data.maxCoeff(), (double) 10.0, (double) 1e-3);
  OpenANN::scaleData(data, (double) -5.0, (double) 5.0);
  ASSERT_EQUALS_DELTA(data.minCoeff(), (double) -5.0, (double) 1e-3);
  ASSERT_EQUALS_DELTA(data.maxCoeff(), (double) 5.0, (double) 1e-3);
  try
  {
    OpenANN::scaleData(data, (double) 1.0, (double) -1.0);
    ASSERT(false);
  }
  catch(OpenANN::OpenANNException& e)
  {
  }
}