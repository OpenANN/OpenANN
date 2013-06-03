#ifndef OPENANN_TEST_DROP_OUT_TEST_CASE_H_
#define OPENANN_TEST_DROP_OUT_TEST_CASE_H_

#include "Test/TestCase.h"

class DropoutTestCase : public TestCase
{
  virtual void run();
  void dropout();
};

#endif // OPENANN_TEST_DROP_OUT_TEST_CASE_H_
