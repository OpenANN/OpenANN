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
  OpenANN::scaleData(data, (fpt) 5.0, (fpt) 10.0);
  ASSERT_EQUALS_DELTA(data.minCoeff(), (fpt) 5.0, (fpt) 1e-3);
  ASSERT_EQUALS_DELTA(data.maxCoeff(), (fpt) 10.0, (fpt) 1e-3);
  OpenANN::scaleData(data, (fpt) -5.0, (fpt) 5.0);
  ASSERT_EQUALS_DELTA(data.minCoeff(), (fpt) -5.0, (fpt) 1e-3);
  ASSERT_EQUALS_DELTA(data.maxCoeff(), (fpt) 5.0, (fpt) 1e-3);
  try
  {
    OpenANN::scaleData(data, (fpt) 1.0, (fpt) -1.0);
    ASSERT(false);
  }
  catch(OpenANN::OpenANNException& e)
  {
  }
}