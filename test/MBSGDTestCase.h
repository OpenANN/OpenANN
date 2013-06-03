#ifndef OPENANN_TEST_MBSGD_TEST_CASE_H_
#define OPENANN_TEST_MBSGD_TEST_CASE_H_

#include <Test/TestCase.h>

class MBSGDTestCase : public TestCase
{
  virtual void run();
  void quadratic();
  void restart();
};

#endif // OPENANN_TEST_MBSGD_TEST_CASE_H_
