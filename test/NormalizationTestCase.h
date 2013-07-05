#ifndef OPENANN_TEST_NORMALIZATION_TEST_CASE_H_
#define OPENANN_TEST_NORMALIZATION_TEST_CASE_H_

#include <Test/TestCase.h>

class NormalizationTestCase : public TestCase
{
  virtual void run();
  void normalize();
};

#endif // OPENANN_TEST_NORMALIZATION_TEST_CASE_H_
