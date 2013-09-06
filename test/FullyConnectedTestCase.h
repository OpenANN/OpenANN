#ifndef OPENANN_TEST_FULLY_CONNECTED_TEST_CASE_H_
#define OPENANN_TEST_FULLY_CONNECTED_TEST_CASE_H_

#include <Test/TestCase.h>

class FullyConnectedTestCase : public TestCase
{
  virtual void run();
  void forward();
  void backprop();
  void inputGradient();
  void parallelForward();
  void regularization();
};

#endif // OPENANN_TEST_FULLY_CONNECTED_TEST_CASE_H_
