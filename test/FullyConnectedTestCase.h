#pragma once

#include <Test/TestCase.h>

class FullyConnectedTestCase : public TestCase
{
  virtual void run();
  void forward();
  void backprop();
  void inputGradient();
  void parallelForward();
};
