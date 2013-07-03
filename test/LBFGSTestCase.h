#ifndef OPENANN_TEST_LBFGS_TEST_CASE_H_
#define OPENANN_TEST_LBFGS_TEST_CASE_H_

#include <Test/TestCase.h>

class LBFGSTestcase : public TestCase
{
public:
  virtual void run();
  void quadratic();
  void restart();
};

#endif // OPENANN_TEST_LBFGS_TEST_CASE_H_
