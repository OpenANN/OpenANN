#ifndef OPENANN_TEST_SIGMA_PI_TEST_CASE_H_
#define OPENANN_TEST_SIGMA_PI_TEST_CASE_H_

#include "Test/TestCase.h"

class SigmaPiTestCase : public TestCase
{
  virtual void run();
  void sigmaPiNoConstraintGradient();
  void sigmaPiWithConstraintGradient();
};

#endif // OPENANN_TEST_SIGMA_PI_TEST_CASE_H_
