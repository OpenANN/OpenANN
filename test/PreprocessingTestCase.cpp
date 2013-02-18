#include "PreprocessingTestCase.h"
#include <Preprocessing.h>
#include <OpenANNException.h>

void PreprocessingTestCase::run()
{
  RUN(PreprocessingTestCase, scaling);
}

void PreprocessingTestCase::scaling()
{
  Mt data = Mt::Random(20, 20);
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