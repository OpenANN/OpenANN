#ifndef OPENANN_TEST_MAX_POOLING_TEST_CASE_H_
#define OPENANN_TEST_MAX_POOLING_TEST_CASE_H_

#include <Test/TestCase.h>

class MaxPoolingTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void maxPooling();
  void maxPoolingGradient();
  void maxPoolingInputGradient();
};

#endif // OPENANN_TEST_MAX_POOLING_TEST_CASE_H_
