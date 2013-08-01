#ifndef OPENANN_TEST_ZCA_TEST_CASE_H_
#define OPENANN_TEST_ZCA_TEST_CASE_H_

#include <Test/TestCase.h>

class ZCATestCase : public TestCase
{
  virtual void run();
  void whiten();
};

#endif // OPENANN_TEST_ZCA_TEST_CASE_H_
