#ifndef OPENANN_TEST_LMA_TEST_CASE_H_
#define OPENANN_TEST_LMA_TEST_CASE_H_

#include <Test/TestCase.h>

class LMATestCase : public TestCase
{
  virtual void run();
  void quadratic();
  void restart();
};

#endif // OPENANN_TEST_LMA_TEST_CASE_H_
