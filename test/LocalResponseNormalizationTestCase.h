#ifndef OPENANN_TEST_LOCAL_RESPONSE_NORMALIZATION_TEST_CASE_H_
#define OPENANN_TEST_LOCAL_RESPONSE_NORMALIZATION_TEST_CASE_H_

#include "Test/TestCase.h"

class LocalResponseNormalizationTestCase : public TestCase
{
  virtual void run();
  void localResponseNormalizationInputGradient();
};

#endif // OPENANN_TEST_LOCAL_RESPONSE_NORMALIZATION_TEST_CASE_H_
