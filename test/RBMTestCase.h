#ifndef OPENANN_TEST_RBM_TEST_CASE_H_
#define OPENANN_TEST_RBM_TEST_CASE_H_

#include <Test/TestCase.h>

class RBMTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void learnSimpleExample();
  void parameterGradient();
  void inputGradient();
};

#endif // OPENANN_TEST_RBM_TEST_CASE_H_
