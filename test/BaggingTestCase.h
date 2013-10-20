#ifndef OPENANN_TEST_BAGGING_TEST_CASE_H_
#define OPENANN_TEST_BAGGING_TEST_CASE_H_

#include <Test/TestCase.h>

class BaggingTestCase : public TestCase
{
  virtual void run();
  virtual void setUp();
  void bagging();
};

#endif // OPENANN_TEST_BAGGING_TEST_CASE_H_
